#!/usr/bin/env python3
"""
LLM authorship attribution — CNN and LSTM classifiers.

Trains two models that identify which LLM *family* produced a piece of text:
  1. TextCNN (Kim 2014): embedding + parallel conv filters (3/4/5-gram) + max-pool
  2. BiLSTM: embedding + bidirectional LSTM + classifier head

Input: the dataset JSON files in files/ (e.g. pollinations_outputs_gpt_oss_20b.json,
llama3.1 outputs). Each record needs: model_family, prompt_id, prompt_category, output.

Important: the train/val/test split is stratified by prompt_id — the same prompt
never appears in more than one split. This forces the classifiers to learn
family writing style instead of memorizing prompt content.

Usage:
    python3 train_authorship_attribution.py                 # train both models
    python3 train_authorship_attribution.py --models cnn    # only the CNN
    python3 train_authorship_attribution.py --epochs 10 --max-len 256

Requires: torch
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "files")
OUT_DIR = os.path.join(HERE, "models")

PAD, UNK = 0, 1


# ---------------------------------------------------------------- data loading

def load_records(data_dir: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*_outputs_*.json"))):
        with open(path) as f:
            data = json.load(f)
        for r in data:
            text = (r.get("output") or "").strip()
            if not text:
                continue  # skip empty outputs
            records.append(
                {
                    "family": r["model_family"],
                    "prompt_id": r["prompt_id"],
                    "category": r.get("prompt_category", "unknown"),
                    "text": text,
                }
            )
    return records


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]", text.lower())


def build_vocab(texts: list[list[str]], min_freq: int = 2) -> dict[str, int]:
    counts = Counter(t for toks in texts for t in toks)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in counts.most_common():
        if c < min_freq:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode(toks: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(t, UNK) for t in toks[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


class AttributionDataset(Dataset):
    def __init__(self, items: list[tuple[list[int], int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        x, y = self.items[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def make_splits(
    records: list[dict],
    families: list[str],
    max_len: int,
    seed: int = 42,
) -> tuple[AttributionDataset, AttributionDataset, AttributionDataset, dict[str, int]]:
    """Split by prompt_id so no prompt leaks across train/val/test.

    Vocab is built from training prompts only, keeping the evaluation honest.
    """
    fam2id = {f: i for i, f in enumerate(families)}
    by_pid: dict[int, list[tuple[str, int]]] = {}
    for r in records:
        by_pid.setdefault(r["prompt_id"], []).append((r["text"], fam2id[r["family"]]))

    pids = sorted(by_pid)
    rnd = random.Random(seed)
    rnd.shuffle(pids)
    n = len(pids)
    tr, va = int(0.8 * n), int(0.9 * n)
    pid_splits = [pids[:tr], pids[tr:va], pids[va:]]

    vocab = build_vocab([tokenize(t) for pid in pid_splits[0] for t, _ in by_pid[pid]])

    def encode_split(chunk) -> AttributionDataset:
        items = [
            (encode(tokenize(t), vocab, max_len), y)
            for pid in chunk
            for t, y in by_pid[pid]
        ]
        return AttributionDataset(items)

    return (*[encode_split(c) for c in pid_splits], vocab)


# -------------------------------------------------------------------- models

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, emb_dim=128, filters=100, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, filters, k) for k in (3, 4, 5)]
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * 3, n_classes)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)  # (B, E, L)
        feats = [F.max_pool1d(F.relu(c(x)), c(x).size(2)).squeeze(2) for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int, emb_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(self.drop(torch.cat([h[0], h[1]], dim=1)))


# ----------------------------------------------------------------- training

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ps += model(x).argmax(1).cpu().tolist()
            ys += y.cpu().tolist()
    acc = sum(a == b for a, b in zip(ys, ps)) / len(ys)
    return acc, ys, ps


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int = 4,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state, stale = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        acc, _, _ = evaluate(model, val_loader, device)
        print(f"  epoch {epoch:2d}: val_acc={acc:.3f}")
        if acc > best_acc + 1e-4:
            best_acc, best_state, stale = acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop (best val_acc={best_acc:.3f})")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def report(name: str, model: nn.Module, test_loader: DataLoader, families: list[str], device: torch.device):
    acc, ys, ps = evaluate(model, test_loader, device)
    print(f"\n{name}: test_acc={acc:.3f}")
    for i, fam in enumerate(families):
        idx = [j for j, y in enumerate(ys) if y == i]
        a = sum(ps[j] == ys[j] for j in idx) / len(idx) if idx else float("nan")
        print(f"  {fam:10s}: {a:.3f}  (n={len(idx)})")
    # confusion matrix
    n = len(families)
    cm = [[0] * n for _ in range(n)]
    for y, p in zip(ys, ps):
        cm[y][p] += 1
    print("  confusion (true x predicted):")
    print("        " + "  ".join(f"{f[:6]:>6}" for f in families))
    for i, fam in enumerate(families):
        print(f"  {fam[:6]:>6} " + "  ".join(f"{cm[i][j]:>6}" for j in range(n)))


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="both", choices=["both", "cnn", "lstm"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records = load_records(DATA_DIR)
    families = sorted({r["family"] for r in records})
    print(f"records: {len(records)}, families: {families}")
    assert len(families) >= 2, "need at least 2 families of outputs to train a classifier"
    for fam in families:
        print(f"  {fam:10s}: {sum(1 for r in records if r['family'] == fam)} outputs")

    train_ds, val_ds, test_ds, vocab = make_splits(records, families, args.max_len, args.seed)
    print(f"vocab size (from train prompts only): {len(vocab)}")
    print(f"split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    loaders = {
        s: DataLoader(ds, batch_size=args.batch_size, shuffle=(s == "train"))
        for s, ds in zip(("train", "val", "test"), (train_ds, val_ds, test_ds))
    }

    want = {"cnn": {"cnn"}, "lstm": {"lstm"}, "both": {"cnn", "lstm"}}[args.models]
    os.makedirs(OUT_DIR, exist_ok=True)
    results: dict = {"families": families, "test_size": len(test_ds)}

    if "cnn" in want:
        model = TextCNN(len(vocab), len(families)).to(device)
        print("\nTraining TextCNN…")
        train_one(model, loaders["train"], loaders["val"], device, args.epochs)
        report("TextCNN", model, loaders["test"], families, device)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "cnn.pt"))
        results["cnn"] = "saved"

    if "lstm" in want:
        model = LSTMClassifier(len(vocab), len(families)).to(device)
        print("\nTraining BiLSTM…")
        train_one(model, loaders["train"], loaders["val"], device, args.epochs)
        report("BiLSTM", model, loaders["test"], families, device)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "lstm.pt"))
        results["lstm"] = "saved"

    torch.save(vocab, os.path.join(OUT_DIR, "vocab.pt"))
    print(f"\nmodels + vocab saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
