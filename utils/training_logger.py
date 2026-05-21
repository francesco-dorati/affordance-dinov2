"""
training_logger.py — Append-only JSONL logger for training metrics.

Each call to .log() writes ONE JSON object on its own line and flushes
immediately. If the process crashes mid-epoch you keep every epoch logged
before the crash. JSONL is trivial to parse: each line is independent JSON,
so you can `tail -1 history.jsonl` from a shell on the remote PC to see the
latest record without any libraries.

Schema (every field is optional — write whatever fields make sense):
    {"timestamp": "...", "epoch": 5, "phase": "train"|"val",
     "loss": float, "loss_mask": float, "loss_normal": float,
     "loss_smooth": float, "iou": float, "angle_deg": float,
     "lr": float, "duration_s": float, "n_batches": int}

A typical training run also writes one-off records:
    {"timestamp": "...", "event": "start", "args": {...}, "n_train": int, ...}
    {"timestamp": "...", "event": "best", "epoch": int, "val_loss": float, ...}
    {"timestamp": "...", "event": "end", "epochs_completed": int, ...}
"""

import json
from datetime import datetime
from pathlib import Path


class JSONLLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, 'a', buffering=1)  # line-buffered

    def log(self, **fields):
        record = {'timestamp': datetime.now().isoformat(timespec='seconds')}
        record.update(fields)
        self._fp.write(json.dumps(record, default=str) + '\n')
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_history(path):
    """Read a JSONL history file into a list of dicts.

    Tolerates a final truncated line caused by a crash.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records
