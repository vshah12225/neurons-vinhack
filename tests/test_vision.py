"""Template matching: the global best vs. the one the plan actually meant.

A cropped template of an empty input box is near-uniform, so a screen with
several identical controls produces one strong correlation peak *per* control.
Keeping only the global maximum is how a re-anchored step landed on a different
box than the planner drew. These tests pin the tie-break: when the caller can
name the point it intended, the nearest near-equal peak wins.
"""

from pathlib import Path

import cv2
import numpy as np

from furti_ai.models import ActionType, Skill
from furti_ai.vision import VisionReflex

#: Deliberately asymmetric so two copies of it correlate identically. A uniform
#: patch would make ``TM_CCOEFF_NORMED`` a 0/0 case.
PATTERN = np.zeros((16, 20, 3), dtype=np.uint8)
for _y in range(PATTERN.shape[0]):
    for _x in range(PATTERN.shape[1]):
        PATTERN[_y, _x] = (30 + 5 * _x, 20 + 8 * _y, 5)

PATTERN_H, PATTERN_W = PATTERN.shape[0], PATTERN.shape[1]
FRAME_H, FRAME_W = 120, 200
TOP_LEFT = (10, 10)
BOTTOM_RIGHT = (150, 90)


def _centre(position: tuple[int, int]) -> tuple[int, int]:
    return (position[0] + PATTERN_W // 2, position[1] + PATTERN_H // 2)


def _frame(*positions: tuple[int, int], inverted: bool = False) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for x, y in positions:
        frame[y : y + PATTERN_H, x : x + PATTERN_W] = PATTERN
    if inverted:
        x, y = BOTTOM_RIGHT
        frame[y : y + PATTERN_H, x : x + PATTERN_W] = PATTERN[::-1, ::-1]
    return frame


class FakeScreen:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def capture(self) -> np.ndarray:
        return self.frame


class RecordingInput:
    def __init__(self):
        self.clicks: list[tuple[int, int]] = []
        self.typed: list[str] = []

    def click(self, x, y, button="left", clicks=1):
        self.clicks.append((x, y))

    def type_text(self, text):
        self.typed.append(text)


def _reflex(frame: np.ndarray, controller=None) -> VisionReflex:
    return VisionReflex(
        FakeScreen(frame),
        controller or RecordingInput(),
        confidence_threshold=0.9,
        multi_scale=False,
    )


# ------------------------------------------------------------------ matching
def test_match_without_a_reference_point_is_the_global_maximum():
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    reflex = _reflex(frame)

    match = reflex._match(frame, PATTERN)

    assert match is not None
    assert match[0] == TOP_LEFT


def test_match_prefers_the_peak_nearest_the_reference_point():
    # Two identical controls, and the plan named the lower one.
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    reflex = _reflex(frame)

    match = reflex._match(frame, PATTERN, near=_centre(BOTTOM_RIGHT))

    assert match is not None
    assert match[0] == BOTTOM_RIGHT


def test_match_keeps_the_global_best_when_it_is_clearly_better():
    # The second location holds a *different* pattern, so its score falls well
    # outside the tie margin and proximity must not promote it.
    frame = _frame(TOP_LEFT, inverted=True)
    reflex = _reflex(frame)

    match = reflex._match(frame, PATTERN, near=_centre(BOTTOM_RIGHT))

    assert match is not None
    assert match[0] == TOP_LEFT


def test_locate_on_passes_the_reference_point_through():
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    reflex = _reflex(frame)

    assert reflex.locate_on(frame, PATTERN)[0] == TOP_LEFT
    assert reflex.locate_on(frame, PATTERN, near=_centre(BOTTOM_RIGHT))[0] == BOTTOM_RIGHT


def test_peaks_are_non_max_suppressed():
    # Without suppression the "second peak" would be a pixel next to the first.
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    reflex = _reflex(frame)
    result = reflex._correlate(frame, PATTERN)

    peaks = reflex._peaks(result, (PATTERN_W, PATTERN_H))

    assert [peak[0] for peak in peaks[:2]] == [TOP_LEFT, BOTTOM_RIGHT]
    assert all(peak[1] > 0.9 for peak in peaks[:2])


# ------------------------------------------------------- stored-anchor replay
def test_expected_center_scales_with_the_frame_size():
    metadata = {
        "expected_bbox": {"x": 50, "y": 40, "width": 20, "height": 16},
        "screen_size": [200, 120],
    }

    assert VisionReflex._expected_center(metadata, (120, 200, 3)) == (60, 48)
    # A doubled capture scales the recorded anchor with it instead of pointing
    # at a stale pixel.
    assert VisionReflex._expected_center(metadata, (240, 400, 3)) == (120, 96)


def test_expected_center_is_none_without_usable_metadata():
    shape = (120, 200, 3)

    assert VisionReflex._expected_center({}, shape) is None
    assert VisionReflex._expected_center(None, shape) is None
    assert VisionReflex._expected_center("nope", shape) is None
    assert VisionReflex._expected_center(
        {"expected_bbox": {"x": 0, "y": 0, "width": 0, "height": 0}}, shape
    ) is None
    assert VisionReflex._expected_center(
        {"expected_bbox": {"x": 1, "y": 1, "width": 4, "height": 4}, "screen_size": [0, 0]},
        shape,
    ) == (3, 3)


def _skill(tmp_path: Path, metadata: dict) -> Skill:
    template = Path(tmp_path) / "duplicate.png"
    cv2.imwrite(str(template), PATTERN)
    return Skill(
        name="duplicate",
        template_path=str(template),
        action=ActionType.CLICK,
        metadata=metadata,
    )


def test_execute_uses_the_stored_anchor_to_pick_the_near_duplicate(tmp_path):
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    controller = RecordingInput()
    reflex = _reflex(frame, controller)
    skill = _skill(
        tmp_path,
        {
            "expected_bbox": {
                "x": BOTTOM_RIGHT[0],
                "y": BOTTOM_RIGHT[1],
                "width": PATTERN_W,
                "height": PATTERN_H,
            },
            "screen_size": [FRAME_W, FRAME_H],
        },
    )

    assert reflex.execute(skill) is True

    assert controller.clicks == [_centre(BOTTOM_RIGHT)]


def test_execute_without_stored_metadata_keeps_the_global_match(tmp_path):
    frame = _frame(TOP_LEFT, BOTTOM_RIGHT)
    controller = RecordingInput()
    reflex = _reflex(frame, controller)

    assert reflex.execute(_skill(tmp_path, {})) is True

    assert controller.clicks == [_centre(TOP_LEFT)]
