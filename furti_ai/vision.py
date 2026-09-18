"""The local execution layer.

``VisionReflex`` replays a compiled skill in milliseconds using OpenCV template
matching -- no network, no tokens. It captures the screen, locates the stored
template with ``cv2.matchTemplate``, and, only if the match confidence clears
the threshold, performs the action. On failure it returns ``False``, which the
orchestrator uses to trigger the ``BrainPlanner`` fallback.

A template of an empty input box or a blank pane is near-uniform, so a screen
with several identical controls produces one strong correlation peak *per*
control and the global maximum may be the wrong one. When the caller knows
where the target was -- a planned ``bbox`` in the executor, the stored
``expected_bbox`` on a reflex replay -- the match nearest that point wins among
peaks that score within :data:`MATCH_TIE_MARGIN` of the best.

Future work: the same ``execute()`` interface can dispatch to a YOLOv8 detector
for the object classes that template matching handles poorly (free-form shapes).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from .controller import InputController
from .models import ActionType, Skill
from .screen import ScreenCapture, map_capture_point_to_input

logger = logging.getLogger(__name__)

# (top-left x, top-left y), confidence, (template width, template height)
MatchResult = tuple[tuple[int, int], float, tuple[int, int]]

#: How many near-equal template locations are inspected when the caller can name
#: the point it meant. Only the global maximum used to be kept, so a second
#: identical control was invisible to the re-anchoring path.
CANDIDATE_PEAKS = 8

#: Peak-to-peak confidence band treated as a tie. ``TM_CCOEFF_NORMED`` varies by
#: a few thousandths between two visually identical controls, so a narrower
#: margin would never see the second one; a much wider one would start
#: preferring genuinely worse matches.
MATCH_TIE_MARGIN = 0.03


def _peak_center(match: MatchResult) -> tuple[int, int]:
    """Centre of a match box, in capture pixels."""
    (x, y), _confidence, (width, height) = match
    return (x + width // 2, y + height // 2)


def _distance_sq(
    left: tuple[int, int], right: tuple[int, int]
) -> int:
    """Squared pixel distance between two points (ordering only)."""
    dx = int(left[0]) - int(right[0])
    dy = int(left[1]) - int(right[1])
    return dx * dx + dy * dy


class VisionReflex:
    """Low-latency template-matching executor."""

    def __init__(
        self,
        screen: ScreenCapture,
        controller: InputController,
        confidence_threshold: float = 0.9,
        multi_scale: bool = True,
        scale_range: tuple[float, float] = (0.5, 1.5),
        scale_steps: int = 11,
    ) -> None:
        self._screen = screen
        self._input = controller
        self.confidence_threshold = confidence_threshold
        self.multi_scale = multi_scale
        self.scale_range = scale_range
        self.scale_steps = scale_steps

    # --------------------------------------------------------------- public
    def execute(self, skill: Skill, variables: Optional[dict[str, Any]] = None) -> bool:
        """Replay a skill against the live screen.

        Returns True if the target was found and the action fired, False
        otherwise (UI moved / changed / threshold not met).
        """
        template = cv2.imread(skill.template_path, cv2.IMREAD_COLOR)
        if template is None:
            logger.error(
                "Template image could not be read for skill %r: %s",
                skill.name,
                skill.template_path,
            )
            return False

        screen_img = self._screen.capture()
        expected = self._expected_center(skill.metadata, screen_img.shape)
        match = self._match(screen_img, template, near=expected)
        if match is None:
            logger.info("No viable template match for skill %r.", skill.name)
            return False

        (x, y), confidence, (tw, th) = match
        if confidence < self.confidence_threshold:
            logger.info(
                "Confidence %.3f below threshold %.2f for skill %r.",
                confidence,
                self.confidence_threshold,
                skill.name,
            )
            return False

        frame_center = (x + tw // 2, y + th // 2)
        cx, cy = self.to_input_point(frame_center, screen_img.shape)
        logger.info(
            "Executing skill %r at frame=(%d, %d), input=(%d, %d) "
            "with confidence %.3f%s.",
            skill.name,
            frame_center[0],
            frame_center[1],
            cx,
            cy,
            confidence,
            f", nearest the stored anchor {tuple(expected)}" if expected else "",
        )
        self._perform_action(skill, (cx, cy), variables or {})
        return True

    @staticmethod
    def _expected_center(
        metadata: Any, frame_shape: tuple[int, ...]
    ) -> Optional[tuple[int, int]]:
        """Where the stored anchor said the target was, in *this* frame's pixels.

        ``expected_bbox`` was recorded on a screen of ``screen_size`` pixels. If
        the capture has changed size since (another monitor, a different DPI
        setting) the box is scaled proportionally before it is compared, so a
        stale size never turns into a bogus reference point. Missing or
        unusable metadata yields ``None`` -- the match then falls back to a
        plain global maximum, exactly as before.
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
            frame_h = int(frame_shape[0]) if len(frame_shape) > 0 else 0
            frame_w = int(frame_shape[1]) if len(frame_shape) > 1 else 0
            if stored_w > 0 and stored_h > 0 and frame_w > 0 and frame_h > 0:
                scale_x = frame_w / stored_w
                scale_y = frame_h / stored_h
        return (
            int(round((x + width / 2) * scale_x)),
            int(round((y + height / 2) * scale_y)),
        )

    def to_input_point(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int]:
        """Map a screenshot-space point to the input backend's coordinates."""
        return map_capture_point_to_input(self._screen, point, frame_shape)

    # ------------------------------------------------------------- matching
    def locate_on(
        self,
        screen_img: np.ndarray,
        template: np.ndarray,
        near: Optional[tuple[int, int]] = None,
        tie_margin: float = MATCH_TIE_MARGIN,
    ) -> Optional[MatchResult]:
        """Best multi-scale match of ``template`` inside ``screen_img``.

        Returns ``(top_left, confidence, (w, h))`` or ``None`` when the
        template is unusable. Exposed for the multi-step executor, which
        re-anchors planned crops on the live screen before acting.

        ``near`` is the capture-space point the caller's plan intended. A crop
        of an empty input box is near-uniform, so several identical boxes can
        score within ``tie_margin`` of each other; the nearest one to ``near``
        then wins instead of the arbitrary global maximum.
        """
        return self._match(screen_img, template, near=near, tie_margin=tie_margin)

    def _match(
        self,
        screen_img: np.ndarray,
        template: np.ndarray,
        near: Optional[tuple[int, int]] = None,
        tie_margin: float = MATCH_TIE_MARGIN,
    ) -> Optional[MatchResult]:
        """Find the best template match, optionally across multiple scales.

        Without ``near`` this is the plain global-argmax search it always was.
        With ``near``, every peak within ``tie_margin`` of the best score is a
        candidate and the one closest to ``near`` wins, so "which of these
        identical controls did the plan mean?" is answered by the plan's own
        geometry rather than by a few thousandths of correlation.
        """
        result = self._correlate(screen_img, template)
        if result is None:
            return None
        best = self._best_of(result, (template.shape[1], template.shape[0]))

        if self.multi_scale and best[1] < self.confidence_threshold:
            low, high = self.scale_range
            for scale in np.linspace(low, high, self.scale_steps):
                if abs(float(scale) - 1.0) < 1e-9:
                    continue
                w = int(round(template.shape[1] * scale))
                h = int(round(template.shape[0] * scale))
                if w < 4 or h < 4:
                    continue
                if w > screen_img.shape[1] or h > screen_img.shape[0]:
                    continue
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(template, (w, h), interpolation=interpolation)
                scaled = self._correlate(screen_img, resized)
                if scaled is None:
                    continue
                candidate = self._best_of(scaled, (w, h))
                if candidate[1] > best[1]:
                    best, result = candidate, scaled

        if near is None:
            return best

        peaks = [
            peak
            for peak in self._peaks(result, (best[2][0], best[2][1]))
            if peak[1] >= best[1] - tie_margin
        ]
        if len(peaks) < 2:
            return best
        return min(peaks, key=lambda peak: _distance_sq(_peak_center(peak), near))

    def _match_once(
        self, screen_img: np.ndarray, template: np.ndarray
    ) -> MatchResult:
        """Single-scale match; returns location, confidence, and template size."""
        result = self._correlate(screen_img, template)
        if result is None:  # pragma: no cover - guarded by the callers
            raise ValueError("template does not fit the screen image")
        return self._best_of(result, (template.shape[1], template.shape[0]))

    @staticmethod
    def _correlate(
        screen_img: np.ndarray, template: np.ndarray
    ) -> Optional[np.ndarray]:
        """Raw ``TM_CCOEFF_NORMED`` map, or ``None`` when the template cannot fit."""
        if (
            template is None
            or template.size == 0
            or screen_img is None
            or template.shape[0] > screen_img.shape[0]
            or template.shape[1] > screen_img.shape[1]
        ):
            return None
        return cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def _best_of(result: np.ndarray, size: tuple[int, int]) -> MatchResult:
        """Strongest location in a correlation map."""
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return (max_loc, float(max_val), size)

    @staticmethod
    def _peaks(
        result: np.ndarray, size: tuple[int, int]
    ) -> list[MatchResult]:
        """The strongest locations in a correlation map, best first.

        Greedy non-max suppression: after taking the maximum, its whole
        template-sized neighbourhood is zeroed so the next iteration finds the
        *next* control rather than a pixel adjacent to the one just taken.
        """
        width, height = int(size[0]), int(size[1])
        work = result.copy()
        peaks: list[MatchResult] = []
        for _ in range(CANDIDATE_PEAKS):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            # A non-positive normalized correlation is not a match at all, and
            # NaN shows up for a zero-variance template.
            if not np.isfinite(max_val) or max_val <= 0.0:
                break
            peaks.append((max_loc, float(max_val), (width, height)))
            x0 = max(0, int(max_loc[0]) - width // 2)
            y0 = max(0, int(max_loc[1]) - height // 2)
            x1 = min(work.shape[1], int(max_loc[0]) + width // 2 + 1)
            y1 = min(work.shape[0], int(max_loc[1]) + height // 2 + 1)
            work[y0:y1, x0:x1] = -1.0
        return peaks

    # ------------------------------------------------------------ actuation
    def _perform_action(
        self,
        skill: Skill,
        center: tuple[int, int],
        variables: Optional[dict[str, Any]] = None,
    ) -> None:
        """Dispatch the skill's action to the input controller."""
        cx, cy = center
        action = skill.action
        metadata = skill.metadata
        variables = variables or {}
        params = metadata.get("params")
        params = params if isinstance(params, dict) else {}

        if action == ActionType.CLICK:
            self._input.click(cx, cy)
        elif action == ActionType.DOUBLE_CLICK:
            self._input.double_click(cx, cy)
        elif action == ActionType.RIGHT_CLICK:
            self._input.right_click(cx, cy)
        elif action == ActionType.DRAG:
            delta = metadata.get("drag_delta")
            if not isinstance(delta, (list, tuple)) or len(delta) != 2:
                raise ValueError("drag reflex has no recorded drag_delta")
            hold_keys = metadata.get("drag_hold_keys")
            if isinstance(hold_keys, str):
                hold_keys = [hold_keys]
            self._input.drag(
                cx,
                cy,
                int(cx + delta[0]),
                int(cy + delta[1]),
                button=str(metadata.get("drag_button") or "left"),
                duration=metadata.get("duration"),
                hold_keys=list(hold_keys) if hold_keys else None,
            )
        elif action == ActionType.TYPE:
            text = str(metadata.get("text") or "")
            for name in metadata.get("reflex_variables", ()):
                token = "{{%s}}" % name
                if token in text:
                    if name not in variables:
                        raise ValueError(f"reflex variable {name!r} was not supplied")
                    text = text.replace(token, str(variables[name]))
            self._input.click(cx, cy)  # focus the field first
            if text:
                self._input.type_text(text)
        elif action == ActionType.SCROLL:
            self._input.click(cx, cy)  # focus the pane first
            self._input.scroll(int(metadata.get("scroll_clicks", 3)))
        elif action == ActionType.KEY_PRESS:
            key = str(metadata.get("key") or params.get("key") or "enter")
            presses = params.get("presses")
            self._input.press_key(
                key, presses=int(presses) if isinstance(presses, (int, float)) else 1
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported action type: {action}")
