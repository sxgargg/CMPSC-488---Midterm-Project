#!/usr/bin/env python3
"""One Qwen worker per machine, dynamic scheduling, one atomic output writer.
Resume this script after interruption: saved prompt IDs may have gaps.
"""
import concurrent.futures as cf
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
MODEL='qwen2.5:7b'
DIGEST='845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e'
HOSTS={'RTX 4070':'http://100.82.174.98:11434','Mac':'http://127.0.0.1:11434'}
OUT=ROOT/'ollama_outputs_qwen2.5_7b.json'
def now():return datetime.now(timezone.utc).isoformat()
def main():
 with (ROOT/'qwen_pc.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  spec=importlib.util.spec_from_file_location('original',ROOT/'collect_ollama_remote.py')
  original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
  prompts=json.loads((ROOT/'llm_classification_prompts_500.json').read_text());by_id={p['id']:p for p in prompts}
  rows,done=original.load_existing(str(OUT));assert len(rows)==len(done)
  for r in rows:
   p=by_id[r['prompt_id']]
   assert r['prompt']==p['prompt'] and r['prompt_category']==p['category'] and r['output'].strip()
   assert r['model_name']==MODEL and r['model_family']=='qwen' and r['temperature']==.7 and r['max_tokens']==512
  for host in HOSTS.values():
   tags=original.api(host,'/api/tags',timeout=15)
   assert next(m for m in tags['models'] if m['name']==MODEL)['digest']==DIGEST
  event={'timestamp_utc':now(),'event':'dual_host_resume','existing_records':len(rows),'workers_per_host':1,'hosts':HOSTS,'model_digest':DIGEST,'temperature':.7,'max_tokens':512,'num_ctx':4096,'existing_dataset_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'reason':'User requested collection on both machines; dynamic scheduling gives faster host more prompts; single writer saves sorted rows.'}
  with (ROOT/'qwen_collection_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  print(f'Resuming across RTX 4070 + Mac: {len(rows)}/500 saved; one worker per host.',flush=True)
  remaining=iter(p for p in prompts if p['id'] not in done)
  def collect(label,p):
   host=HOSTS[label];start=time.monotonic()
   payload={'model':MODEL,'prompt':p['prompt'],'stream':False,'options':{'temperature':.7,'num_predict':512,'num_ctx':4096}}
   def request():
    r=original.api(host,'/api/generate',payload)
    if not r.get('done') or not r.get('response','').strip():raise RuntimeError('Incomplete or empty response')
    return r
   response=original.with_retry(request);elapsed=time.monotonic()-start
   row={'model_family':'qwen','model_name':MODEL,'prompt_id':p['id'],'prompt_category':p['category'],'prompt':p['prompt'],'output':response['response'].strip(),'temperature':.7,'max_tokens':512,'timestamp_utc':now(),'ollama_host':host}
   metrics={k:response.get(k) for k in ('done_reason','eval_count','eval_duration','prompt_eval_count','prompt_eval_duration','total_duration','load_duration')}
   metrics.update(prompt_id=p['id'],timestamp_utc=now(),wall_seconds=elapsed,host=host,worker=label)
   return row,metrics
  with cf.ThreadPoolExecutor(max_workers=2) as pool:
   pending={}
   for label in HOSTS:
    p=next(remaining,None)
    if p is not None:pending[pool.submit(collect,label,p)]=label
   while pending:
    finished,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
    for future in finished:
     label=pending.pop(future);row,metrics=future.result()
     assert row['prompt_id'] not in done
     rows.append(row);rows.sort(key=lambda r:r['prompt_id']);original.save_rows(str(OUT),rows);done.add(row['prompt_id'])
     with (ROOT/'qwen_split_metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
     print(f"  [{len(rows)}/500 saved] prompt {row['prompt_id']}: {metrics['eval_count']} tokens, {metrics['wall_seconds']:.2f}s ({label})",flush=True)
     time.sleep(.5)
     p=next(remaining,None)
     if p is not None:pending[pool.submit(collect,label,p)]=label
  assert len(rows)==500 and done==set(range(1,501))
  print('Done: all 500 Qwen responses saved, unique and sorted.',flush=True)
if __name__=='__main__':main()
