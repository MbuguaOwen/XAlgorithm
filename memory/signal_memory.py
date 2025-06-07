"""Signal event memory for performance tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque, Dict, Any, Optional

from .memory_core import MemoryCore


@dataclass
class SignalRecord:
    timestamp: datetime
    features: Dict[str, Any]
    outcome: Optional[Dict[str, Any]] = None
    confidence: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class SignalMemory:
    """Circular memory of signal outcomes."""

    def __init__(self, memory: MemoryCore, maxlen: int = 1000) -> None:
        self.memory = memory
        if not isinstance(self.memory.signal_memory, deque) or self.memory.signal_memory.maxlen != maxlen:
            self.memory.signal_memory = deque(self.memory.signal_memory, maxlen=maxlen)

    def log_signal(self, record: SignalRecord) -> None:
        self.memory.signal_memory.append(record.to_dict())
        self.memory.save_all()

    def recall(self) -> Deque[dict]:
        return self.memory.signal_memory


__all__ = ["SignalRecord", "SignalMemory"]
