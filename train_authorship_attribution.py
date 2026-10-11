#!/usr/bin/env python3
"""Reproducible CNN/LSTM attribution; explicit files, grouped splits, saved evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
FILES = {'llama': 'ollama_outputs_llama3.1_8b.json', 'gemini': 'gemini_outputs_gemini_3.8_flash.jsonl', 'qwen': 'ollama_outputs_qwen2.5_7b.json'}
FAMILIES = sorted(FILES)
TARGET_CATEGORIES = ['creative_fiction', 'poetry', 'marketing_copy', 'business_writing']
NAME_RE = re.compile(r'\b(?:qwen|gemini|llama|chatgpt|openai|anthropic|deepseek|claude|gpt)(?:[ .-]?\d+(?:\.\d+)*)?\b', re.I)
TOKEN_RE = re.compile(r'<model>|\n|[^\W_]+(?:[’\'][^\W_]+)*|[^\w\s]', re.UNICODE)


def load_data(root=ROOT):
    prompts = json.loads((root / 'llm_classification_prompts_500.json').read_text())
    master = {p['id']: p for p in prompts}
    assert len(master) == 500 and set(master) == set(range(1, 501))
    assert sorted(Counter(p['category'] for p in prompts).values()) == [25] * 20
    rows = []
    for family, name in FILES.items():
        text = (root / name).read_text()
        data = [json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(text)
        assert len(data) == 500 and sorted(r['prompt_id'] for r in data) == list(range(1, 501))
        for r in data:
            p = master[r['prompt_id']]
            assert r['model_family'] == family and r['prompt'] == p['prompt'] and r['prompt_category'] == p['category']
            assert r['temperature'] == .7 and r['max_tokens'] == 512 and r['output'].strip()
        rows.extend(data)
    return sorted(rows, key=lambda r: (r['prompt_id'], r['model_family']))


def make_split(rows, domain=False, seed=42):
    categories = defaultdict(set)
    for r in rows:
        categories[r['prompt_category']].add(r['prompt_id'])
    split = {'train': [], 'val': [], 'test': []}
    if domain:
        split['target'] = []
    rng = random.Random(seed)
    for i, category in enumerate(sorted(categories)):
        ids = sorted(categories[category]); rng.shuffle(ids)
        if domain and category in TARGET_CATEGORIES:
            split['target'].extend(ids)
        else:
            # Alternate 2/3 validation and 3/2 test per category: 400/50/50 overall.
            nval = 2 + (sum(len(v) for k,v in split.items() if k != 'target') // 25) % 2
            split['train'].extend(ids[:20]); split['val'].extend(ids[20:20+nval]); split['test'].extend(ids[20+nval:])
    split = {k: sorted(v) for k,v in split.items()}
    seen = set()
    for ids in split.values():
        assert not (seen & set(ids)); seen.update(ids)
    assert seen == set(range(1,501))
    return split


def tokenize(text, transform='masked'):
    if transform != 'raw':
        text = NAME_RE.sub('<model>', text)
    if transform == 'plain':
        # Deliberately removes punctuation/newlines and capitalization; retains lexical words.
        return re.findall(r'[^\W_]+', text.lower(), re.UNICODE)
    return TOKEN_RE.findall(text.lower())


def row_tokens(row, source, transform='masked', max_output=256, max_input=64):
    prompt = tokenize(row['prompt'], transform)[:max_input]
    output = tokenize(row['output'], transform)[:max_output]
    return {'input': prompt, 'output': output, 'both': prompt + ['<sep>'] + output}[source] or ['<empty>']


def build_vocab(token_lists, min_freq=2):
    counts = Counter(t for ts in token_lists for t in ts)
    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2, '<empty>': 3}
    for token, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if n >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def collate(batch):
    xs, ys, pids = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    x = torch.zeros(len(xs), max(5, int(lengths.max())), dtype=torch.long)
    for i, ids in enumerate(xs):
        x[i, :len(ids)] = torch.tensor(ids)
    return x, lengths, torch.tensor(ys), torch.tensor(pids)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=128, width=64, dropout=.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, width, k) for k in (3,4,5)])
        self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(width*3, 3)
    def forward(self, x, lengths):
        h = self.embedding(x).transpose(1,2); features = []
        for conv in self.convs:
            z = torch.relu(conv(h)); valid = (lengths-conv.kernel_size[0]+1).clamp(min=1)
            mask = torch.arange(z.shape[2], device=x.device)[None,:] >= valid[:,None]
            features.append(z.masked_fill(mask[:,None,:], -1e9).amax(2))
        return self.fc(self.dropout(torch.cat(features,1)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=128, width=128, dropout=.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        # Unidirectional recurrence: gather at last REAL token, before any right padding.
        self.lstm = nn.LSTM(emb, width, batch_first=True)
        self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(width,3)
    def forward(self, x, lengths):
        h,_ = self.lstm(self.embedding(x))
        last = h[torch.arange(x.shape[0],device=x.device), lengths-1]
        return self.fc(self.dropout(last))


def metrics(y, pred):
    cm = confusion_matrix(y,pred,labels=list(range(3)))
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=list(range(3)),average='macro',zero_division=0)), 'per_family_accuracy':{f:float(cm[i,i]/cm[i].sum()) for i,f in enumerate(FAMILIES)}, 'confusion_matrix':cm.tolist(), 'n':len(y)}


def evaluate(model, loader, device):
    model.eval(); y=[]; pred=[]; pids=[]; losses=[]
    with torch.no_grad():
        for x,lengths,labels,ids in loader:
            logits=model(x.to(device),lengths.to(device)); losses.append(nn.functional.cross_entropy(logits,labels.to(device),reduction='sum').item())
            pred.extend(logits.argmax(1).cpu().tolist()); y.extend(labels.tolist()); pids.extend(ids.tolist())
    return metrics(y,pred), [{'prompt_id':p,'true':FAMILIES[a],'predicted':FAMILIES[b]} for p,a,b in zip(pids,y,pred)],sum(losses)/len(y)


def bootstrap_ci(predictions, seed=42, repeats=2000):
    grouped=defaultdict(list)
    for r in predictions: grouped[r['prompt_id']].append(r['true']==r['predicted'])
    scores=np.array([np.mean(v) for v in grouped.values()]);rng=np.random.default_rng(seed)
    means=scores[rng.integers(len(scores),size=(repeats,len(scores)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def run(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.device=='auto':
        args.device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device=torch.device(args.device)
    rows=load_data();split=make_split(rows,args.domain)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() and not args.overwrite: raise RuntimeError(f'{out} already has results; use a fresh --out')
    tokens=[row_tokens(r,args.text_source,args.transform,args.max_output,args.max_input) for r in rows]
    train_ids=set(split['train']);vocab=build_vocab([ts for r,ts in zip(rows,tokens) if r['prompt_id'] in train_ids])
    loaders={}; truncation={}
    for part,ids in split.items():
        subset=[(r,ts) for r,ts in zip(rows,tokens) if r['prompt_id'] in set(ids)]
        items=[([vocab.get(t,1) for t in ts],FAMILIES.index(r['model_family']),r['prompt_id']) for r,ts in subset]
        loaders[part]=DataLoader(items,batch_size=args.batch_size,shuffle=part=='train',collate_fn=collate,generator=torch.Generator().manual_seed(args.seed))
        truncation[part]={'output_exceeds_limit':sum(len(tokenize(r['output'],args.transform))>args.max_output for r,_ in subset),'input_exceeds_limit':sum(len(tokenize(r['prompt'],args.transform))>args.max_input for r,_ in subset),'unknown_token_fraction':sum(t not in vocab for _,ts in subset for t in ts)/sum(len(ts) for _,ts in subset),'n':len(subset)}
    model=(TextCNN if args.model=='cnn' else LSTMClassifier)(len(vocab)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    history=[];best_loss=float('inf');stale=0;start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        t=time.perf_counter();model.train();loss_total=0;n=0
        for x,lengths,y,_ in loaders['train']:
            optimizer.zero_grad(); logits=model(x.to(device),lengths.to(device));loss=nn.functional.cross_entropy(logits,y.to(device));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_total+=loss.item()*len(y);n+=len(y)
        vm,_,vl=evaluate(model,loaders['val'],device)
        history.append({'epoch':epoch,'train_loss':loss_total/n,'val_loss':vl,'val_accuracy':vm['accuracy'],'seconds':time.perf_counter()-t})
        print(f'{out.name} epoch {epoch}: train_loss={loss_total/n:.4f} val_loss={vl:.4f} val_acc={vm["accuracy"]:.3f} ({history[-1]["seconds"]:.1f}s)',flush=True)
        if vl < best_loss-1e-4:
            best_loss=vl;best_epoch=epoch;stale=0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=args.patience:break
    model.load_state_dict(best)
    result={'config':vars(args),'families':FAMILIES,'vocab_size':len(vocab),'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'history':history,'training_seconds':time.perf_counter()-start,'preprocessing':truncation,'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform()},'data_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES.values()}}
    for part in ['val','test']+(['target'] if args.domain else []):
        m,predictions,loss=evaluate(model,loaders[part],device);m['loss']=loss;m['accuracy_prompt_bootstrap_95ci']=bootstrap_ci(predictions);result[part]=m
        (out/f'{part}_predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'split.json').write_text(json.dumps(split,indent=2)+'\n');(out/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False)+'\n')
    checkpoint=ROOT/'models'/out.name;checkpoint.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':best,'vocab':vocab,'config':vars(args),'families':FAMILIES},checkpoint/'model.pt')
    print(json.dumps({'run':out.name,'test':result['test'],'seconds':result['training_seconds']}),flush=True)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['cnn','lstm'],required=True);p.add_argument('--text-source',choices=['input','output','both'],default='output');p.add_argument('--transform',choices=['raw','masked','plain'],default='masked');p.add_argument('--domain',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--patience',type=int,default=4);p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-output',type=int,default=256);p.add_argument('--max-input',type=int,default=64);p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto');p.add_argument('--threads',type=int,default=4);p.add_argument('--out',required=True);p.add_argument('--overwrite',action='store_true');return p
if __name__=='__main__':run(parser().parse_args())
