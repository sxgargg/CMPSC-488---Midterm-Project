#!/usr/bin/env python3
"""Classify supplied text using a locally trained checkpoint."""
import argparse,json
from pathlib import Path
import torch
from train_authorship_attribution import TextCNN,LSTMClassifier,row_tokens,collate
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',default='');p.add_argument('--output-file',type=Path);args=p.parse_args()
state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);c=state['config']
text=args.output_file.read_text() if args.output_file else ''
if c['text_source']!='input' and not text.strip():p.error('--output-file is required for output/both models')
if c['text_source']!='output' and not args.prompt.strip():p.error('--prompt is required for input/both models')
ts=row_tokens({'prompt':args.prompt,'output':text},c['text_source'],c['transform'],c['max_output'],c['max_input']);v=state['vocab'];x,lengths,_,_=collate([([v.get(t,1) for t in ts],0,0)])
model=(TextCNN if c['model']=='cnn' else LSTMClassifier)(len(v));model.load_state_dict(state['state_dict']);model.eval()
with torch.no_grad():scores=model(x,lengths).softmax(1)[0].tolist()
print(json.dumps({'prediction':state['families'][scores.index(max(scores))],'uncalibrated_probabilities':dict(zip(state['families'],scores))},indent=2))
