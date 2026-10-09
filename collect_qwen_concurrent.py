#!/usr/bin/env python3
"""Two-request Qwen collection; resumes existing JSON and commits in prompt order.
Uses an isolated local Ollama service configured with two slots. Sampling stays
at temperature 0.7, num_predict 512; no seed override. Original collector unchanged.
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from collections import deque

ROOT = Path(__file__).resolve().parent
MODEL = 'qwen2.5:7b'
HOST = 'http://127.0.0.1:11435'
OUT = ROOT / 'ollama_outputs_qwen2.5_7b.json'

def main():
    spec = importlib.util.spec_from_file_location('original_collector', ROOT / 'collect_ollama_remote.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    prompts = json.loads((ROOT / 'llm_classification_prompts_500.json').read_text())
    by_id = {p['id']:p for p in prompts}
    rows, done = original.load_existing(str(OUT))
    assert len(done) == len(rows), 'Duplicate IDs in existing data'
    for row in rows:
        assert row['model_name'] == MODEL and row['model_family'] == 'qwen'
        assert row['temperature'] == 0.7 and row['max_tokens'] == 512
        assert row['prompt'] == by_id[row['prompt_id']]['prompt'] and row['output'].strip()
    remaining = [p for p in prompts if p['id'] not in done]
    if not remaining:
        print('Done: all 500 prompts already saved.', flush=True)
        return
    env = os.environ.copy()
    env.update(OLLAMA_HOST='127.0.0.1:11435', OLLAMA_NUM_PARALLEL='2')
    with (ROOT / 'qwen_concurrent_server.log').open('a') as server_log:
        server = subprocess.Popen(['/usr/local/bin/ollama','serve'], env=env, stdout=server_log, stderr=server_log, start_new_session=True)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError('Temporary Ollama server exited; see server log')
                try:
                    urllib.request.urlopen(HOST + '/api/version', timeout=1).close()
                    break
                except Exception:
                    time.sleep(.2)
            else:
                raise RuntimeError('Local concurrency server did not start')
            event = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'event':'concurrent_resume', 'existing_records':len(rows), 'workers':2, 'model':MODEL, 'temperature':0.7, 'max_tokens':512, 'host':HOST, 'existing_dataset_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest() if OUT.exists() else None}
            with (ROOT / 'qwen_collection_events.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            print(f'\nResuming Qwen with 2 concurrent requests: {len(rows)}/500 saved. Committing in prompt order.', flush=True)
            def collect(p):
                text = original.with_retry(lambda: original.generate(HOST, MODEL, p['prompt'], .7, 512))
                if not text.strip():
                    raise RuntimeError(f"Empty output for prompt {p['id']}; stopped without marking complete")
                return {'model_family':'qwen','model_name':MODEL,'prompt_id':p['id'],'prompt_category':p['category'],'prompt':p['prompt'],'output':text.strip(),'temperature':.7,'max_tokens':512,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'ollama_host':HOST}
            pending = deque()
            iterator = iter(remaining)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(2):
                    p = next(iterator, None)
                    if p is not None: pending.append(pool.submit(collect,p))
                while pending:
                    row = pending.popleft().result()
                    rows.append(row)
                    original.save_rows(str(OUT),rows)
                    print(f"  [{row['prompt_id']}/500] {len(row['output'])} chars (2 workers)", flush=True)
                    time.sleep(.5)
                    p = next(iterator,None)
                    if p is not None: pending.append(pool.submit(collect,p))
            assert sorted(r['prompt_id'] for r in rows) == list(range(1,501))
            print(f'Done: {OUT} (500 rows)', flush=True)
        finally:
            try:
                request = urllib.request.Request(HOST+'/api/generate',data=json.dumps({'model':MODEL,'keep_alive':0}).encode(),headers={'Content-Type':'application/json'})
                urllib.request.urlopen(request,timeout=30).close()
            except Exception:
                pass
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                server.wait(timeout=15)

if __name__ == '__main__':
    main()
