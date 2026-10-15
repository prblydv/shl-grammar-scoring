"""Progress events written by the workload; independent liveness is in runner.py."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

_writer = None
_context = {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def context(**values):
    _context.update(values)


def emit(stage, **values):
    value = dict(_context, **values, stage=stage, updated_at=time.time(),
                 timestamp=datetime.now(timezone.utc).isoformat())
    directory = os.environ.get('SHL_RUN_DIR')
    if directory:
        directory = Path(directory)
        atomic_json(directory / 'progress.json', value)
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
    return value


def scalar(name, value, step):
    global _writer
    directory = os.environ.get('SHL_RUN_DIR')
    if directory and _writer is None:
        from torch.utils.tensorboard import SummaryWriter
        _writer = SummaryWriter(str(Path(directory) / 'tensorboard'))
    if _writer:
        _writer.add_scalar(name, value, step)
        _writer.flush()


def close():
    global _writer
    if _writer:
        _writer.close()
        _writer = None


@contextmanager
def run_lock(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.run.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is already writing to this run directory')
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
