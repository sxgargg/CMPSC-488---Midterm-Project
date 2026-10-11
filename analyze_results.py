#!/usr/bin/env python3
"""Summarize saved predictions, descriptive features, and diagnostic baselines."""
import csv,json,re
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train_authorship_attribution import ROOT,FAMILIES,load_data,make_split,metrics,bootstrap_ci,NAME_RE

FEATURES=['words','characters','lines','blank_lines','bullet_lines','heading_lines','backticks','bold_markers','type_token_ratio','mean_word_length','punctuation_fraction']
def features(text):
    words=re.findall(r'[^\W_]+',text.lower());lines=text.splitlines()
    return [len(words),len(text),len(lines),sum(not x.strip() for x in lines),sum(bool(re.match(r'^\s*(?:[-*+] |\d+[.)] )',x)) for x in lines),sum(bool(re.match(r'^\s*#{1,6} ',x)) for x in lines),text.count('`'),text.count('**'),len(set(words))/max(1,len(words)),np.mean([len(x) for x in words]) if words else 0,sum(not x.isalnum() and not x.isspace() for x in text)/max(1,len(text))]

def paired_ci(a,b):
    aa={(r['prompt_id'],r['true']):r['true']==r['predicted'] for r in a};bb={(r['prompt_id'],r['true']):r['true']==r['predicted'] for r in b};assert aa.keys()==bb.keys()
    ids=sorted({x[0] for x in aa});d=np.array([np.mean([int(bb[(pid,f)])-int(aa[(pid,f)]) for f in FAMILIES]) for pid in ids]);rng=np.random.default_rng(42)
    ci=np.quantile(d[rng.integers(len(d),size=(5000,len(d)))].mean(1),[.025,.975])
    return {'difference_b_minus_a':float(d.mean()),'prompt_bootstrap_95ci':ci.tolist()}

def main():
    rows=load_data();results=ROOT/'results';fig=ROOT/'figures';fig.mkdir(exist_ok=True)
    split=make_split(rows);x=np.array([features(NAME_RE.sub('<model>',r['output'])) for r in rows]);y=np.array([FAMILIES.index(r['model_family']) for r in rows]);train=np.array([r['prompt_id'] in split['train'] for r in rows]);test=np.array([r['prompt_id'] in split['test'] for r in rows])
    descriptive={f:{name:{'mean':float(x[y==i,j].mean()),'median':float(np.median(x[y==i,j]))} for j,name in enumerate(FEATURES)} for i,f in enumerate(FAMILIES)}
    baselines={}
    for name,cols in [('length_only',[0]),('structure_all',list(range(len(FEATURES))))]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=42));model.fit(x[train][:,cols],y[train]);pred=model.predict(x[test][:,cols]);m=metrics(y[test],pred)
        records=[{'prompt_id':r['prompt_id'],'true':FAMILIES[a],'predicted':FAMILIES[b]} for r,a,b in zip([r for r in rows if r['prompt_id'] in split['test']],y[test],pred)]
        m['accuracy_prompt_bootstrap_95ci']=bootstrap_ci(records);m['features']=[FEATURES[c] for c in cols];m['standardized_coefficients']=model[-1].coef_.tolist();baselines[name]=m
        (results/f'{name}_predictions.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=[];runs={}
    for path in sorted(results.glob('*/metrics.json')):
        r=json.loads(path.read_text());runs[path.parent.name]=r
        for part in ['test']+(['target'] if 'target' in r else []):
            m=r[part];summary.append({'run':path.parent.name,'evaluation':part,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'n':m['n'],'best_epoch':r['best_epoch'],'training_seconds':r['training_seconds'],**m['per_family_accuracy']})
    with (results/'summary.csv').open('w') as f:
        if summary:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    comparisons={}
    for model in ['cnn','lstm']:
        a=results/f'{model}_output_s42'/'test_predictions.json'
        for tag,other in [('adding_prompt',f'{model}_both_s42'),('removing_formatting',f'{model}_plain_s42')]:
            b=results/other/'test_predictions.json'
            if a.exists() and b.exists():comparisons[f'{model}_{tag}']=paired_ci(json.loads(a.read_text()),json.loads(b.read_text()))
    audit={'descriptive':descriptive,'feature_names':FEATURES,'diagnostic_baselines':baselines,'paired_comparisons':comparisons,'split':split}
    (results/'analysis.json').write_text(json.dumps(audit,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if all(f'{m}_{s}_s42' in runs for m in ['cnn','lstm'] for s in ['input','output','both']):
        f,ax=plt.subplots(figsize=(7.5,3.6));xx=np.arange(3)
        for i,(model,color) in enumerate([('cnn','#167D9A'),('lstm','#DF8745')]):
            vals=[runs[f'{model}_{s}_s42']['test']['accuracy'] for s in ['input','output','both']];ax.bar(xx+(i-.5)*.32,vals,.32,label=model.upper(),color=color)
        ax.axhline(1/3,color='gray',ls='--',label='Chance');ax.set(xticks=xx,xticklabels=['Input only','Output only','Input + output'],ylabel='Test accuracy',ylim=(0,1));ax.legend(ncol=3,loc='upper left');f.tight_layout();f.savefig(fig/'rq2_accuracy.png',dpi=180);plt.close(f)
        f,axs=plt.subplots(1,2,figsize=(7.5,3.4))
        for ax,model in zip(axs,['cnn','lstm']):
            cm=np.array(runs[f'{model}_output_s42']['test']['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=50)
            for i in range(3):
                for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>28 else 'black')
            ax.set(xticks=range(3),yticks=range(3),xticklabels=FAMILIES,yticklabels=FAMILIES,title=model.upper(),xlabel='Predicted',ylabel='True')
        f.tight_layout();f.savefig(fig/'confusion_matrices.png',dpi=180);plt.close(f)
    f,axs=plt.subplots(1,3,figsize=(8,3.2))
    for ax,name in zip(axs,['words','bold_markers','bullet_lines']):
        ax.boxplot([x[y==i,FEATURES.index(name)] for i in range(3)],tick_labels=FAMILIES,showfliers=False);ax.set_title(name.replace('_',' ').title());ax.tick_params(axis='x',rotation=20)
    f.tight_layout();f.savefig(fig/'style_features.png',dpi=180);plt.close(f)
    print(json.dumps({'runs':len(runs),'baselines':baselines,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
