# tests/ghosts/

Recorded keypoint traces of things that are **not people**, kept so a fix can
be shown to reject them and a later change can be shown not to re-admit them.

Deliberately NOT in `tests/captures/`: that directory is auto-discovered by
`test_captured.py`, which asserts the recorded gesture *fires*. A false
positive filed there would turn the bug into a requirement.

| file | what it is |
|---|---|
| `painting_wall_art_*.npz` | A framed figure on the living-room wall, detected at YOLO confidence 0.71 — above the strict acquire threshold, so unreachable by any confidence gate. Confirmed human, then fired FOLDED_ARMS 17 times in 20 minutes from one long-lived pid (774) in an empty room, 2026-09-27. |

These hold keypoints only, not foreground masks, so they prove the negative
side of a liveness test (a static region scores zero) and cannot prove the
positive side. See `tests/test_body_liveness.py`.
