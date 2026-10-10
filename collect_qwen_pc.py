#!/usr/bin/env python3
"""Resume Qwen sequentially on the verified RTX 4070, preserving saved rows."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
HOST = 'http://100.82.174.98:11434'
MODEL = 'qwen2.5:7b'
DIGEST = '845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e'
OUT = ROOT / 'ollama_outputs_qwen2.5_7b.json'

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    with (ROOT / 'qwen_pc.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = importlib.util.spec_from_file_location('original', ROOT / 'collect_ollama_remote.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        prompts = json.loads((ROOT / 'llm_classification_prompts_500.json').read_text())
        by_id = {p['id']: p for p in prompts}
        rows, done = original.load_existing(str(OUT))
        assert [r['prompt_id'] for r in rows] == [p['id'] for p in prompts[:len(rows)]]
        for r in rows:
            p = by_id[r['prompt_id']]
            assert r['prompt'] == p['prompt'] and r['prompt_category'] == p['category']
            assert r['model_name'] == MODEL and r['model_family'] == 'qwen'
            assert r['temperature'] == .7 and r['max_tokens'] == 512 and r['output'].strip()
        tags = original.api(HOST, '/api/tags', timeout=15)
        assert next(m for m in tags['models'] if m['name'] == MODEL)['digest'] == DIGEST
        event = {'timestamp_utc': now(), 'event': 'remote_sequential_resume', 'existing_records': len(rows), 'workers': 1, 'model': MODEL, 'model_digest': DIGEST, 'host': HOST, 'temperature': .7, 'max_tokens': 512, 'num_ctx': 4096, 'existing_dataset_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(), 'reason': 'RTX 4070 handoff: one slot had faster and more stable benchmark throughput'}
        with (ROOT / 'qwen_collection_events.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
        print(f'Resuming on RTX 4070: {len(rows)}/500 saved, 1 request at a time.', flush=True)
        for p in prompts:
            if p['id'] in done:
                continue
            payload = {'model': MODEL, 'prompt': p['prompt'], 'stream': False, 'options': {'temperature': .7, 'num_predict': 512, 'num_ctx': 4096}}
            def request():
                response = original.api(HOST, '/api/generate', payload)
                if not response.get('done') or not response.get('response', '').strip():
                    raise RuntimeError('Incomplete or empty response')
                return response
            started = time.monotonic()
            response = original.with_retry(request)
            elapsed = time.monotonic() - started
            row = {'model_family': 'qwen', 'model_name': MODEL, 'prompt_id': p['id'], 'prompt_category': p['category'], 'prompt': p['prompt'], 'output': response['response'].strip(), 'temperature': .7, 'max_tokens': 512, 'timestamp_utc': now(), 'ollama_host': HOST}
            rows.append(row)
            original.save_rows(str(OUT), rows)
            metrics = {k: response.get(k) for k in ('done_reason', 'eval_count', 'eval_duration', 'prompt_eval_count', 'prompt_eval_duration', 'total_duration', 'load_duration')}
            metrics.update(prompt_id=p['id'], timestamp_utc=now(), wall_seconds=elapsed, host=HOST)
            with (ROOT / 'qwen_pc_metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics) + '\n')
            print(f"  [{p['id']}/500] {len(row['output'])} chars, {response.get('eval_count')} tokens, {elapsed:.2f}s (RTX 4070)", flush=True)
            time.sleep(.5)
        assert len(rows) == 500 and {r['prompt_id'] for r in rows} == set(range(1, 501))
        print('Done: all 500 Qwen responses saved.', flush=True)

if __name__ == '__main__':
    main()
