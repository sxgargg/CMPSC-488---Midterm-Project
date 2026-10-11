#!/usr/bin/env python3
"""Recompute reported metrics and check every experiment against its saved split."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from train_authorship_attribution import ROOT,FAMILIES,FILES,load_data,make_split,metrics
from run_experiments import grid
rows=load_data();baseline=make_split(rows)
for name,model,source,transform,seed,domain in grid():
    path=ROOT/'results'/name;r=json.loads((path/'metrics.json').read_text());s=json.loads((path/'split.json').read_text())
    assert s==make_split(rows,domain)
    best_loss=float('inf');best_epoch=None
    for epoch in r['history']:
        if epoch['val_loss'] < best_loss-1e-4:
            best_loss=epoch['val_loss'];best_epoch=epoch['epoch']
    assert r['best_epoch']==best_epoch
    for file,h in r['data_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h
    train=[x for x in rows if x['prompt_id'] in s['train']]
    from train_authorship_attribution import row_tokens,build_vocab
    assert json.loads((path/'vocab.json').read_text())==build_vocab([row_tokens(x,source,transform) for x in train])
    for part in ['val','test']+(['target'] if domain else []):
        ps=json.loads((path/f'{part}_predictions.json').read_text())
        assert len(ps)==3*len(s[part]);assert Counter((x['prompt_id'],x['true']) for x in ps)==Counter((pid,f) for pid in s[part] for f in FAMILIES)
        check=metrics([FAMILIES.index(x['true']) for x in ps],[FAMILIES.index(x['predicted']) for x in ps])
        for key in ['accuracy','macro_f1','per_family_accuracy','confusion_matrix','n']:assert check[key]==r[part][key],(name,part,key)
        if source=='input':assert abs(check['accuracy']-1/3)<1e-10
    print('Verified',name)
print('PASS: all 14 runs, data hashes, train-only vocabularies, splits, checkpoint selection, and reported metrics.')
