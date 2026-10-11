#!/usr/bin/env python3
"""Execute the predeclared experiment grid; completed runs are verified and skipped."""
import argparse,json,subprocess,sys
from pathlib import Path
from train_authorship_attribution import ROOT

def grid():
    for model in ['cnn','lstm']:
        for source in ['input','output','both']:
            yield f'{model}_{source}_s42',model,source,'masked',42,False
        for seed in [7,123]:
            yield f'{model}_output_s{seed}',model,'output','masked',seed,False
        yield f'{model}_plain_s42',model,'output','plain',42,False
        yield f'{model}_domain_s42',model,'output','masked',42,True

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='auto');args=p.parse_args()
    for name,model,source,transform,seed,domain in grid():
        out=ROOT/'results'/name
        if (out/'metrics.json').exists():
            c=json.loads((out/'metrics.json').read_text())['config']
            assert (c['model'],c['text_source'],c['transform'],c['seed'],c['domain'])==(model,source,transform,seed,domain)
            print('Already complete:',name,flush=True);continue
        cmd=[sys.executable,str(ROOT/'train_authorship_attribution.py'),'--model',model,'--text-source',source,'--transform',transform,'--seed',str(seed),'--device',args.device,'--out',str(out)]
        if domain:cmd.append('--domain')
        subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
