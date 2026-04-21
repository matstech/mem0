import logging
from typing import Dict, Optional


def _pluralize(label: str) -> str:
    if label.endswith("s"):
        return label
    return f"{label}s"


class Mem0BenchmarkProgress:
    """Structured progress logger for Mem0 benchmarks."""

    def __init__(self, benchmark_name: str, *, enabled: bool = False, mode: str = "detailed"):
        self.benchmark_name = benchmark_name
        self.enabled = enabled
        self.mode = mode
        self.logger = logging.getLogger(f"evaluation.memzero.{benchmark_name.replace(':', '.')}")
        self.total_conversations = 0
        self.total_units: Optional[int] = None
        self.run_unit_label = "unit"
        self.completed_conversations = 0
        self.completed_units = 0
        self.current_conversation_index: Optional[int] = None
        self.current_conversation_id: Optional[str] = None
        self.current_total_units: Optional[int] = None
        self.current_unit_label = "unit"

    @property
    def tqdm_disabled(self) -> bool:
        return self.enabled

    def start_run(
        self,
        *,
        total_conversations: int,
        total_units: Optional[int] = None,
        unit_label: str = "unit",
        note: Optional[str] = None,
    ) -> None:
        self.total_conversations = int(total_conversations)
        self.total_units = int(total_units) if total_units is not None else None
        self.run_unit_label = unit_label
        self.completed_conversations = 0
        self.completed_units = 0
        self.current_conversation_index = None
        self.current_conversation_id = None
        self.current_total_units = None
        self.current_unit_label = unit_label
        self._emit(phase="setup", step="run_start", note=note)

    def start_conversation(
        self,
        conversation_index: int,
        *,
        conversation_id: Optional[str] = None,
        total_units: Optional[int] = None,
        unit_label: str = "unit",
        note: Optional[str] = None,
    ) -> None:
        self.current_conversation_index = int(conversation_index)
        self.current_conversation_id = conversation_id
        self.current_total_units = int(total_units) if total_units is not None else None
        self.current_unit_label = unit_label
        self._emit(
            phase="setup",
            step="conversation_start",
            completed_current_units=0,
            note=note,
        )

    def log_step(
        self,
        *,
        phase: str,
        step: str,
        current_unit: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        completed_current_units = None
        if current_unit is not None:
            completed_current_units = max(int(current_unit) - 1, 0)
        self._emit(
            phase=phase,
            step=step,
            active_unit=current_unit,
            completed_current_units=completed_current_units,
            note=note,
            extra=extra,
        )

    def mark_unit_complete(
        self,
        *,
        phase: str,
        step: str,
        current_unit: int,
        note: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        self.completed_units += 1
        self._emit(
            phase=phase,
            step=step,
            active_unit=current_unit,
            completed_current_units=int(current_unit),
            note=note,
            extra=extra,
        )

    def finish_conversation(
        self,
        *,
        unit_count: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        completed_current_units = self.current_total_units
        if unit_count is not None:
            self.completed_units += int(unit_count)
        self.completed_conversations += 1
        self._emit(
            phase="complete",
            step="conversation_complete",
            completed_current_units=completed_current_units,
            note=note,
            extra=extra,
        )
        self.current_conversation_index = None
        self.current_conversation_id = None
        self.current_total_units = None

    def mark_conversation_complete(
        self,
        *,
        conversation_index: Optional[int] = None,
        conversation_id: Optional[str] = None,
        unit_count: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if conversation_index is not None:
            self.current_conversation_index = int(conversation_index)
        if conversation_id is not None:
            self.current_conversation_id = conversation_id
        if unit_count is not None:
            self.completed_units += int(unit_count)
        self.completed_conversations += 1
        self._emit(
            phase="complete",
            step="conversation_complete",
            completed_current_units=None,
            note=note,
            extra=extra,
        )
        self.current_conversation_index = None
        self.current_conversation_id = None
        self.current_total_units = None

    def finish_run(self, *, note: Optional[str] = None) -> None:
        self._emit(phase="complete", step="run_complete", note=note)

    def _conversation_progress(self, completed_current_units: Optional[int]) -> Optional[float]:
        if self.current_total_units is None:
            return None
        if self.current_total_units <= 0:
            return 100.0
        if completed_current_units is None:
            completed_current_units = 0
        return min(max((float(completed_current_units) / float(self.current_total_units)) * 100.0, 0.0), 100.0)

    def _overall_progress(self, completed_current_units: Optional[int]) -> float:
        if self.total_units:
            return min(max((float(self.completed_units) / float(self.total_units)) * 100.0, 0.0), 100.0)

        if self.total_conversations <= 0:
            return 100.0

        current_fraction = 0.0
        if self.current_total_units and completed_current_units is not None and self.current_total_units > 0:
            current_fraction = float(completed_current_units) / float(self.current_total_units)
        return min(
            max(((self.completed_conversations + current_fraction) / float(self.total_conversations)) * 100.0, 0.0),
            100.0,
        )

    def _remaining_conversations(self) -> int:
        if self.current_conversation_index is not None:
            return max(self.total_conversations - (self.current_conversation_index + 1), 0)
        return max(self.total_conversations - self.completed_conversations, 0)

    def _emit(
        self,
        *,
        phase: str,
        step: str,
        active_unit: Optional[int] = None,
        completed_current_units: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.enabled:
            return

        fields = {
            "phase": phase,
            "step": step,
            "mode": self.mode,
        }

        if self.total_conversations:
            fields["completed_conversations"] = f"{self.completed_conversations}/{self.total_conversations}"
            fields["remaining_conversations"] = self._remaining_conversations()

        overall_progress = self._overall_progress(completed_current_units)
        fields["overall_progress"] = f"{overall_progress:.1f}%"

        if self.current_conversation_index is not None and self.total_conversations:
            fields["conversation"] = f"{self.current_conversation_index + 1}/{self.total_conversations}"
            if self.current_conversation_id is not None:
                fields["conversation_id"] = self.current_conversation_id

        conversation_progress = self._conversation_progress(completed_current_units)
        if conversation_progress is not None:
            fields["conv_progress"] = f"{conversation_progress:.1f}%"

        if active_unit is not None and self.current_total_units is not None:
            fields[self.current_unit_label] = f"{active_unit}/{self.current_total_units}"

        if self.current_total_units is not None and completed_current_units is not None:
            remaining_current_units = max(self.current_total_units - completed_current_units, 0)
            fields[f"remaining_{_pluralize(self.current_unit_label)}"] = remaining_current_units

        if self.total_units is not None:
            remaining_total_units = max(self.total_units - self.completed_units, 0)
            fields[f"remaining_total_{_pluralize(self.run_unit_label)}"] = int(remaining_total_units)

        if note:
            fields["note"] = note

        if extra:
            for key, value in extra.items():
                fields[key] = value

        message = " ".join(f"{key}={value}" for key, value in fields.items())
        self.logger.info("[%s] %s", self.benchmark_name, message)
