"""Re-anchoring a stored reflex that no longer matches the screen.

A compiled reflex is a cropped template plus metadata. It is fast, cheap and
completely blind: when the UI moves, it simply fails to match, and the old
behaviour was to throw the whole reflex away and ask the planner to compile a
new one from scratch.

:class:`ReflexRealigner` is the middle step the pipeline was missing. It hands
the failing reflex *back to the LLM* -- "this element, where is it now?" -- and
recompiles the stored template in place, so a reflex that drifted (a window
moved, the list scrolled, the theme changed) is repaired instead of discarded.
Only when re-alignment also fails does the orchestrator fall back to full
re-planning.

A reflex that keeps failing is retired (see :meth:`ReflexRealigner.retire`):
keeping a reflex that never matches costs a screen capture and a template match
on every single run, which is worse than having no cache entry at all.
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .jsoncontract import extract_json_object
from .models import Skill

logger = logging.getLogger(__name__)

__all__ = ["REALIGN_SYSTEM_PROMPT", "RealignResult", "ReflexRealigner"]

#: A re-located anchor further than this fraction of the screen diagonal from
#: the stored one is not "the same element, moved" -- it is a different control.
#: On a screen with two identical input boxes that is exactly how a wrong answer
#: used to be accepted and then written over a good stored template, turning a
#: transient mismatch into a permanent mis-click. The orchestrator falls back to
#: re-planning when this refuses, so refusing is always safe.
REALIGN_DRIFT_REJECT_FRACTION = 0.5

#: Above this fraction of the diagonal the move is plausible but large (a window
#: dragged, a list scrolled, a panel resized), so the re-location has to be
#: confident as well as plausible.
REALIGN_DRIFT_WARN_FRACTION = 0.25

#: Confidence a large-drift re-alignment must reach, on top of
#: ``reflex_min_anchor_confidence``.
REALIGN_DRIFT_MIN_CONFIDENCE = 0.9

REALIGN_SYSTEM_PROMPT = (
    "You are the reflex re-alignment component of Furti AI, a desktop "
    "automation agent. A previously compiled reflex (a cropped target image) "
    "no longer matches the screen, so its stored anchor is stale. Look at the "
    "current screenshot and find the SAME element again: the window may have "
    "moved, the list may have scrolled, or the theme may have changed.\n"
    "Return JSON only, exactly this shape (one bare object, no fences, no "
    "prose, integer bbox pixels):\n"
    '{"found": true|false, '
    '"bbox": {"x": 0, "y": 0, "width": 0, "height": 0}, '
    '"confidence": 0.0, '
    '"description": "<what you located>", '
    '"note": "<what changed, or why it cannot be found>"}\n'
    "The bbox is in pixels relative to the top-left of the screenshot and must "
    "tightly surround the element. Set found=false (with confidence 0) when the "
    "element is genuinely not on screen -- do not guess a location you cannot "
    "see."
)


def _previous_center(
    metadata: Any, frame_width: int, frame_height: int
) -> Optional[tuple[int, int]]:
    """The stored anchor's centre in *this* frame's pixels, when it is usable.

    ``expected_bbox`` was recorded on a screen of ``screen_size`` pixels, so it
    is scaled by the frame-size ratio before use. An anchor that then falls
    outside the frame describes some other screen (another monitor, a stale
    size, hand-written metadata), so it is ignored rather than trusted -- a
    guard that refused a re-alignment on unusable evidence would block a
    legitimate repair.
    """
    if not isinstance(metadata, dict):
        return None
    bbox = metadata.get("expected_bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    scale_x = scale_y = 1.0
    size = metadata.get("screen_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            stored_w, stored_h = int(size[0]), int(size[1])
        except (TypeError, ValueError):
            stored_w = stored_h = 0
        if stored_w > 0 and stored_h > 0:
            scale_x = frame_width / stored_w
            scale_y = frame_height / stored_h

    center_x = int(round((x + width / 2) * scale_x))
    center_y = int(round((y + height / 2) * scale_y))
    if not (0 <= center_x <= frame_width and 0 <= center_y <= frame_height):
        return None
    return center_x, center_y


def _drift_fraction(
    metadata: Any,
    bbox: tuple[int, int, int, int],
    frame_shape: Any,
) -> Optional[float]:
    """How far ``bbox`` sits from the stored anchor, as a share of the diagonal.

    ``None`` when the reflex carries no usable previous anchor: there is then
    nothing to compare against and the guard does not apply.
    """
    try:
        frame_height = int(frame_shape[0])
        frame_width = int(frame_shape[1])
    except (IndexError, TypeError, ValueError):
        return None
    if frame_width <= 0 or frame_height <= 0:
        return None
    previous = _previous_center(metadata, frame_width, frame_height)
    if previous is None:
        return None
    diagonal = math.hypot(frame_width, frame_height)
    if diagonal <= 0:
        return None
    current_x = bbox[0] + bbox[2] / 2
    current_y = bbox[1] + bbox[3] / 2
    return math.hypot(current_x - previous[0], current_y - previous[1]) / diagonal


@dataclass(frozen=True)
class RealignResult:
    """Outcome of one re-alignment attempt."""

    ok: bool = False
    skill: Optional[Skill] = None
    reason: str = ""
    confidence: float = 0.0
    retired: bool = False

    def note(self) -> str:
        if self.retired:
            return f"reflex retired: {self.reason}"
        if self.ok and self.skill is not None:
            return (
                f"reflex {self.skill.name!r} re-aligned on the live screen "
                f"(confidence {self.confidence:.2f})"
            )
        return f"re-alignment failed: {self.reason}"


class ReflexRealigner:
    """Re-locates a stale reflex with the LLM and rewrites its template."""

    def __init__(
        self,
        settings: Any,
        llm: Any,
        memory: Any,
        screen: Any = None,
        journal: Any = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._memory = memory
        self._screen = screen
        self._journal = journal

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return bool(
            getattr(self._settings, "reflex_realign", True)
        ) and self._llm is not None

    def failure_limit(self) -> int:
        return int(getattr(self._settings, "reflex_retire_failures", 3) or 0)

    def is_useless(self, skill: Skill) -> bool:
        """True when this reflex has missed often enough to be dropped."""
        limit = self.failure_limit()
        return bool(limit) and int(getattr(skill, "failure_count", 0)) >= limit

    def retire(self, skill: Skill, reason: str = "") -> RealignResult:
        """Forget a reflex that has stopped being useful."""
        forget = getattr(self._memory, "forget", None)
        if callable(forget):
            forget(skill.name)
        detail = reason or (
            f"it failed {skill.failure_count} replays without matching"
        )
        self._warn(
            f"Retired reflex {skill.name!r}: {detail}. It will be compiled "
            "again from scratch if the task runs once more."
        )
        return RealignResult(ok=False, reason=detail, retired=True)

    # --------------------------------------------------------------- realign
    def realign(self, skill: Skill, failure_reason: str = "") -> RealignResult:
        """Ask the LLM where ``skill``'s target is now, then rewrite its template."""
        if not self.enabled:
            return RealignResult(ok=False, reason="re-alignment is disabled")
        if self._screen is None:
            return RealignResult(ok=False, reason="no screen capture backend")
        try:
            frame = self._screen.capture()
        except Exception as exc:  # noqa: BLE001 - capture can fail headless
            return RealignResult(ok=False, reason=f"screen capture failed: {exc}")
        if frame is None or getattr(frame, "size", 0) == 0:
            return RealignResult(ok=False, reason="screen capture returned no frame")

        self._thought(
            f"Reflex {skill.name!r} no longer matches the screen "
            f"({failure_reason or 'template match failed'}); asking the model "
            "to re-align it."
        )
        user = self._prompt(skill, frame, failure_reason)
        try:
            raw = self._llm.chat_vision(
                REALIGN_SYSTEM_PROMPT,
                user,
                _encode_png(frame),
                purpose="reflex_realign",
            )
            payload = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001 - the planner is the fallback
            self._warn(f"Reflex re-alignment call failed: {exc}")
            return RealignResult(ok=False, reason=str(exc))

        output = getattr(self._journal, "ai_output", None)
        if callable(output):
            output(raw, getattr(self._llm, "_model", "model"), "reflex_realign")

        bbox = _bbox_from_payload(payload.get("bbox"))
        confidence = _as_float(payload.get("confidence"))
        if not _truthy(payload.get("found")):
            reason = str(payload.get("note") or "the model could not find the element")
            self._warn(f"Reflex {skill.name!r} could not be re-aligned: {reason}")
            return RealignResult(ok=False, reason=reason)
        if bbox is None:
            return RealignResult(ok=False, reason="the model returned no usable bbox")
        if confidence is None:
            # Overwriting a stored template on an unquantified guess is how a
            # good reflex becomes a bad one; demand a number.
            return RealignResult(
                ok=False, reason="the re-location reported no confidence score"
            )
        threshold = float(
            getattr(self._settings, "reflex_min_anchor_confidence", 0.75)
        )
        if confidence < threshold:
            return RealignResult(
                ok=False,
                reason=(
                    f"re-location confidence {confidence:.2f} is below "
                    f"reflex_min_anchor_confidence={threshold:.2f}"
                ),
            )

        # A plausible-looking answer is not enough: the model may have found a
        # *different* control that looks the same. Without this check the new
        # anchor is written over the stored one for every later replay.
        drift = _drift_fraction(skill.metadata, bbox, frame.shape)
        if drift is not None:
            if drift > REALIGN_DRIFT_REJECT_FRACTION:
                reason = (
                    f"the re-located element sits {drift:.0%} of the screen "
                    "diagonal from its previous position, which is a different "
                    "control rather than the same one moved"
                )
                self._warn(f"Reflex {skill.name!r} re-alignment refused: {reason}.")
                return RealignResult(ok=False, reason=reason)
            if drift > REALIGN_DRIFT_WARN_FRACTION:
                required = max(threshold, REALIGN_DRIFT_MIN_CONFIDENCE)
                if confidence < required:
                    reason = (
                        f"the re-located element moved {drift:.0%} of the screen "
                        f"diagonal and its confidence {confidence:.2f} is below "
                        f"the {required:.2f} such a move requires"
                    )
                    self._warn(
                        f"Reflex {skill.name!r} re-alignment refused: {reason}."
                    )
                    return RealignResult(ok=False, reason=reason)
                self._warn(
                    f"Reflex {skill.name!r} re-aligned {drift:.0%} of the screen "
                    "diagonal away from its previous position."
                )

        crop, error = _crop(frame, bbox)
        if crop is None:
            return RealignResult(ok=False, reason=error or "the bbox was not usable")

        try:
            path = self._write_template(skill, crop)
        except OSError as exc:
            return RealignResult(ok=False, reason=f"could not save the template: {exc}")

        previous = skill.metadata.get("expected_bbox")
        skill.metadata["expected_bbox"] = {
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
        }
        skill.metadata["screen_size"] = [int(frame.shape[1]), int(frame.shape[0])]
        skill.metadata["anchor"] = f"realigned conf={confidence:.2f}"
        skill.metadata["realign_note"] = str(payload.get("note", "")).strip()
        skill.metadata["realign_of"] = previous
        if drift is not None:
            skill.metadata["realign_drift"] = round(drift, 3)
        skill.record_realign(str(path))
        save = getattr(self._memory, "save_skill", None)
        if callable(save):
            save(skill)
        self._journal_note(
            f"Re-aligned reflex {skill.name!r} -> {Path(path).name} "
            f"(confidence {confidence:.2f}, was {previous}"
            f"{f', drift {drift:.0%}' if drift is not None else ''})"
        )
        return RealignResult(ok=True, skill=skill, confidence=confidence)

    # -------------------------------------------------------------- plumbing
    def _prompt(self, skill: Skill, frame: np.ndarray, failure_reason: str) -> str:
        metadata = skill.metadata or {}
        height, width = int(frame.shape[0]), int(frame.shape[1])
        lines = [
            f"Reflex name: {skill.name}",
            f"Action it performs: {skill.action.value}",
            f"Original description: {metadata.get('description', '(not recorded)')}",
        ]
        if metadata.get("target"):
            lines.append(f"Original target label: {metadata['target']}")
        if metadata.get("expected_bbox"):
            lines.append(f"Last known bounding box: {metadata['expected_bbox']}")
        if metadata.get("anchor"):
            lines.append(f"Original anchor: {metadata['anchor']}")
        if skill.realign_count:
            lines.append(f"Times already re-aligned: {skill.realign_count}")
        lines.extend(
            [
                f"Failure: {failure_reason or 'the stored template did not match'}",
                f"Screenshot size: {width}x{height}",
                "",
                "Find that same element in the screenshot and return its bbox now.",
            ]
        )
        return "\n".join(lines)

    def _write_template(self, skill: Skill, crop: np.ndarray) -> Path:
        """Overwrite the reflex's template file with the re-located crop."""
        path = Path(skill.template_path)
        if not path.suffix:
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"cv2 could not write {path}")
        return path

    def _warn(self, message: str) -> None:
        warn = getattr(self._journal, "warn", None)
        if callable(warn):
            warn(message)

    def _thought(self, message: str) -> None:
        thought = getattr(self._journal, "thought", None)
        if callable(thought):
            thought(message)

    def _journal_note(self, message: str) -> None:
        reflex = getattr(self._journal, "reflex", None)
        if callable(reflex):
            reflex(message)


# ------------------------------------------------------------------ helpers
def _encode_png(image: np.ndarray) -> str:
    """Base64 PNG for the vision API."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode the screenshot as PNG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _parse_json(raw: str) -> dict[str, Any]:
    """Read the re-alignment answer through the shared strict contract reader."""
    payload = extract_json_object(raw)
    if not isinstance(payload, dict):  # pragma: no cover - reader guarantees this
        raise ValueError("the re-alignment response was not a JSON object")
    return payload


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_from_payload(value: Any) -> Optional[tuple[int, int, int, int]]:
    """Read ``(x, y, w, h)`` from the shapes a model emits."""
    if isinstance(value, dict):
        try:
            return (
                int(round(float(value["x"]))),
                int(round(float(value["y"]))),
                int(round(float(value["width"]))),
                int(round(float(value["height"]))),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(round(float(item))) for item in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _crop(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[Optional[np.ndarray], str]:
    """Clamp a bbox to the frame and crop it, rejecting junk geometry."""
    x, y, width, height = bbox
    frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    if width < 4 or height < 4:
        return None, f"the returned bbox is too small ({width}x{height})"
    if width * height > 0.6 * frame_width * frame_height:
        return None, "the returned bbox covers most of the screen"
    return frame[y : y + height, x : x + width], ""
