"""
Track stitching: merge fragmented tracks that belong to the same player.

ByteTrack assigns a fresh ID whenever it briefly loses a player (e.g. when
one player stands behind another). This post-processor reconnects those
fragments: when one track ends and another track starts shortly afterwards
nearby, both are treated as the same physical player and collapsed onto a
single ID.

The link-finding here is the canonical implementation; the evaluation
module reuses it to report fragmentation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MAX_FRAME_GAP = 30
DEFAULT_MAX_DISTANCE_PX = 80.0


@dataclass(slots=True)
class TrackEndpoints:
    """First and last observed position of a single track."""

    track_id: int
    start_frame: int
    end_frame: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float


def _track_endpoints(df: pd.DataFrame) -> list[TrackEndpoints]:

    endpoints: list[TrackEndpoints] = []

    for track_id, track in df.sort_values("frame").groupby("track_id"):

        first = track.iloc[0]
        last = track.iloc[-1]

        endpoints.append(
            TrackEndpoints(
                track_id=int(track_id),
                start_frame=int(first["frame"]),
                end_frame=int(last["frame"]),
                start_x=(first["x1"] + first["x2"]) / 2,
                start_y=(first["y1"] + first["y2"]) / 2,
                end_x=(last["x1"] + last["x2"]) / 2,
                end_y=(last["y1"] + last["y2"]) / 2,
            )
        )

    return endpoints


def find_continuation_links(
    df: pd.DataFrame,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> dict[int, int]:
    """
    Find tracks that are most likely the resumption of an ended track.

    Returns a mapping {continuation_track_id: ended_track_id}. For each
    ended track, the closest track that starts within ``max_frame_gap``
    frames and ``max_distance_px`` pixels is greedily claimed as its
    continuation. Each track can only continue one earlier track.
    """

    endpoints = _track_endpoints(df)
    endpoints.sort(key=lambda t: t.end_frame)

    used_as_continuation: set[int] = set()
    links: dict[int, int] = {}

    for ended in endpoints:

        best_candidate: int | None = None
        best_distance = max_distance_px

        for started in endpoints:

            if started.track_id == ended.track_id:
                continue

            if started.track_id in used_as_continuation:
                continue

            frame_gap = started.start_frame - ended.end_frame

            if not (0 < frame_gap <= max_frame_gap):
                continue

            distance = float(
                np.hypot(
                    started.start_x - ended.end_x,
                    started.start_y - ended.end_y,
                )
            )

            if distance < best_distance:
                best_distance = distance
                best_candidate = started.track_id

        if best_candidate is not None:
            used_as_continuation.add(best_candidate)
            links[best_candidate] = ended.track_id

    return links


def _resolve_root(track_id: int, links: dict[int, int]) -> int:
    """Follow a continuation chain back to its originating track id."""

    seen: set[int] = set()

    while track_id in links and track_id not in seen:
        seen.add(track_id)
        track_id = links[track_id]

    return track_id


def build_id_remapping(links: dict[int, int]) -> dict[int, int]:
    """
    Collapse continuation chains so every fragment maps to the id of the
    first track in its chain (e.g. C->B->A becomes {C: A, B: A}).
    """

    return {track_id: _resolve_root(track_id, links) for track_id in links}


def stitch_tracks(
    df: pd.DataFrame,
    class_name: str = "person",
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with fragmented ``class_name`` tracks merged
    onto a single id per physical player.
    """

    persons = df[df["class_name"] == class_name].dropna(subset=["track_id"]).copy()
    persons["track_id"] = persons["track_id"].astype(int)

    links = find_continuation_links(persons, max_frame_gap, max_distance_px)
    remapping = build_id_remapping(links)

    result = df.copy()
    has_track = result["track_id"].notna()

    result.loc[has_track, "track_id"] = (
        result.loc[has_track, "track_id"].astype(int).replace(remapping)
    )

    return result


def stitch_csv(
    input_csv: Path,
    output_csv: Path,
    class_name: str = "person",
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> None:
    """Read a detections CSV, stitch its tracks and write the result."""

    df = pd.read_csv(input_csv)

    stitched = stitch_tracks(df, class_name, max_frame_gap, max_distance_px)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(output_csv, index=False)

    before = df.dropna(subset=["track_id"])["track_id"].nunique()
    after = stitched.dropna(subset=["track_id"])["track_id"].nunique()

    print(f"Stitched tracks: {before} -> {after} unique IDs")
    print(f"Wrote {output_csv}")
