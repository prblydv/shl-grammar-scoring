"""Supervised parent: timestamped stdout, independent heartbeats and writer lock."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from telemetry import atomic_json, run_lock


def gpu_metrics():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return [dict(zip(['utilization_percent','used_mib','total_mib'], map(float, line.split(','))))
                    for line in result.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def snapshot(directory, pid, state='running'):
    directory = Path(directory)
    try:
        progress = json.loads((directory / 'progress.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {'stage':'starting', 'updated_at':time.time()}
    age = max(0, time.time()-progress['updated_at'])
    return dict(progress, process_state=state, pid=pid, heartbeat_at=time.time(),
                timestamp=datetime.now(timezone.utc).isoformat(), seconds_since_progress=age,
                stalled_warning=state == 'running' and age >= 300,
                gpu=gpu_metrics(), free_disk_gib=shutil.disk_usage(directory).free/2**30)


def monitor(command, directory, interval=15):
    directory = Path(directory).resolve()
    with run_lock(directory):
        environment = dict(os.environ, SHL_RUN_DIR=str(directory), SHL_MANAGED_RUN='1', PYTHONUNBUFFERED='1')
        atomic_json(directory / 'progress.json', {'stage':'starting', 'updated_at':time.time()})
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=environment)
        log_lock = threading.Lock()
        def log(line):
            stamped = f'[{datetime.now(timezone.utc).isoformat()}] {line.rstrip()}'
            with log_lock:
                with (directory / 'train.log').open('a') as stream:
                    stream.write(stamped+'\n')
                print(stamped, flush=True)
        def read_output():
            for line in process.stdout:
                log(line)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                status = snapshot(directory, process.pid)
                atomic_json(directory / 'latest_status.json', status)
                with (directory / 'heartbeats.jsonl').open('a') as stream:
                    stream.write(json.dumps(status)+'\n')
                log('HEARTBEAT '+json.dumps(status))
                try:
                    process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
            reader.join(timeout=10)
            state = 'complete' if process.returncode == 0 else 'failed'
            status = snapshot(directory, process.pid, state)
            status['exit_code'] = process.returncode
            atomic_json(directory / 'latest_status.json', status)
            atomic_json(directory / 'terminal_status.json', status)
            if process.returncode:
                atomic_json(directory / 'error.json', {'exit_code':process.returncode,
                    'message':'Workload failed; traceback/details are in train.log', 'timestamp':status['timestamp']})
            log('TERMINAL '+json.dumps(status))
            return process.returncode
        except BaseException:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            atomic_json(directory / 'terminal_status.json', snapshot(directory, process.pid, 'interrupted'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('first_run.json'))
    args = parser.parse_args()
    command = [sys.executable, '-u', str(Path(__file__).with_name('pipeline.py')), 'run',
               '--output', str(args.output.resolve()), '--config', str(args.config.resolve())]
    config=json.loads(args.config.read_text())
    interval=config.get('heartbeat_seconds',15)
    if not isinstance(interval,(int,float)) or not 0<interval<=60: raise ValueError('Heartbeat interval must be in (0,60] seconds')
    sys.exit(monitor(command, args.output, interval=interval))


if __name__ == '__main__': main()
