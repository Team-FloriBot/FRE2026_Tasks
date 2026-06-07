#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from task4.coverage_planner import Point2D, Waypoint, normalize_polygon_points


@dataclass(frozen=True)
class ShotTarget:
    target_id: int
    x: float
    y: float
    z: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class ShootingPlannerConfig:
    shooting_range_m: float = 2.0
    shoot_angle_min_deg: float = -60.0
    shoot_angle_max_deg: float = 60.0
    candidate_grid_spacing_m: float = 0.5
    yaw_sample_step_deg: float = 10.0
    path_candidate_stride_m: float = 0.5
    object_ring_distance_ratio: float = 0.75
    object_ring_angle_step_deg: float = 20.0


@dataclass(frozen=True)
class ShootingPose:
    x: float
    y: float
    yaw: float
    target_ids: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class _Candidate:
    x: float
    y: float
    yaw: float
    covered_ids: frozenset


def plan_shooting_poses(
    targets: Sequence[ShotTarget],
    polygon_points: Sequence[Point2D],
    coverage_path: Sequence[Waypoint],
    start_xy: Point2D,
    config: ShootingPlannerConfig,
) -> Tuple[List[ShootingPose], List[ShotTarget]]:
    polygon = normalize_polygon_points(polygon_points)
    unique_targets = _unique_targets_inside_polygon(targets, polygon)
    if not unique_targets:
        return [], []

    candidates = _build_candidates(unique_targets, polygon, coverage_path, start_xy, config)
    scored_candidates = _score_candidates(candidates, unique_targets, config)

    target_by_id = {target.target_id: target for target in unique_targets}
    uncovered_ids: Set[int] = set(target_by_id)
    selected: List[ShootingPose] = []
    last_xy = (float(start_xy[0]), float(start_xy[1]))

    while uncovered_ids:
        best = None
        best_gain: Set[int] = set()
        best_score = None

        for candidate in scored_candidates:
            gain = set(candidate.covered_ids) & uncovered_ids
            if not gain:
                continue

            travel_distance = distance_2d(last_xy, (candidate.x, candidate.y))
            score = (len(gain), len(candidate.covered_ids), -travel_distance)
            if best_score is None or score > best_score:
                best = candidate
                best_gain = gain
                best_score = score

        if best is None:
            break

        selected.append(
            ShootingPose(
                x=best.x,
                y=best.y,
                yaw=best.yaw,
                target_ids=sorted(best_gain),
            )
        )
        uncovered_ids -= best_gain
        last_xy = (best.x, best.y)

    uncovered = [target_by_id[target_id] for target_id in sorted(uncovered_ids)]
    return selected, uncovered


def _unique_targets_inside_polygon(
    targets: Sequence[ShotTarget],
    polygon: Sequence[Point2D],
) -> List[ShotTarget]:
    by_id: Dict[int, ShotTarget] = {}
    for target in targets:
        if point_in_polygon((target.x, target.y), polygon):
            by_id[int(target.target_id)] = target
    return list(by_id.values())


def _build_candidates(
    targets: Sequence[ShotTarget],
    polygon: Sequence[Point2D],
    coverage_path: Sequence[Waypoint],
    start_xy: Point2D,
    config: ShootingPlannerConfig,
) -> List[Point2D]:
    candidates: Dict[Tuple[int, int], Point2D] = {}

    def add_candidate(x: float, y: float) -> None:
        point = (float(x), float(y))
        if point_in_polygon(point, polygon):
            key = (round(point[0] * 1000), round(point[1] * 1000))
            candidates[key] = point

    add_candidate(start_xy[0], start_xy[1])

    stride = max(0.05, float(config.path_candidate_stride_m))
    last_added = None
    for waypoint in coverage_path:
        point = (waypoint.x, waypoint.y)
        if last_added is None or distance_2d(last_added, point) >= stride:
            add_candidate(point[0], point[1])
            last_added = point

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)
    spacing = max(0.05, float(config.candidate_grid_spacing_m))

    x = min_x
    while x <= max_x + 1e-9:
        y = min_y
        while y <= max_y + 1e-9:
            add_candidate(x, y)
            y += spacing
        x += spacing

    ring_distance = max(
        0.05,
        float(config.shooting_range_m) * max(0.05, min(1.0, config.object_ring_distance_ratio)),
    )
    angle_step = math.radians(max(1.0, float(config.object_ring_angle_step_deg)))
    for target in targets:
        angle = 0.0
        while angle < 2.0 * math.pi:
            add_candidate(
                target.x - ring_distance * math.cos(angle),
                target.y - ring_distance * math.sin(angle),
            )
            angle += angle_step

    return list(candidates.values())


def _score_candidates(
    candidate_positions: Sequence[Point2D],
    targets: Sequence[ShotTarget],
    config: ShootingPlannerConfig,
) -> List[_Candidate]:
    scored = []
    yaw_step = math.radians(max(1.0, float(config.yaw_sample_step_deg)))

    for x, y in candidate_positions:
        yaw = 0.0
        while yaw < 2.0 * math.pi:
            covered = set()
            for target in targets:
                if target_reachable((x, y), yaw, target, config):
                    covered.add(int(target.target_id))
            if covered:
                scored.append(_Candidate(x=x, y=y, yaw=normalize_angle(yaw), covered_ids=frozenset(covered)))
            yaw += yaw_step

    return scored


def target_reachable(
    shooter_xy: Point2D,
    shooter_yaw: float,
    target: ShotTarget,
    config: ShootingPlannerConfig,
) -> bool:
    dx = float(target.x) - float(shooter_xy[0])
    dy = float(target.y) - float(shooter_xy[1])
    if math.hypot(dx, dy) > max(0.0, float(config.shooting_range_m)):
        return False

    world_bearing = math.atan2(dy, dx)
    relative_bearing = normalize_angle(world_bearing - float(shooter_yaw))
    return angle_in_window(
        relative_bearing,
        math.radians(float(config.shoot_angle_min_deg)),
        math.radians(float(config.shoot_angle_max_deg)),
    )


def angle_in_window(angle: float, min_angle: float, max_angle: float) -> bool:
    raw_width = abs(float(max_angle) - float(min_angle))
    if raw_width >= 2.0 * math.pi:
        return True

    angle = normalize_positive(angle)
    min_angle = normalize_positive(min_angle)
    max_angle = normalize_positive(max_angle)

    if min_angle <= max_angle:
        return min_angle <= angle <= max_angle

    return angle >= min_angle or angle <= max_angle


def point_in_polygon(point: Point2D, polygon_points: Sequence[Point2D]) -> bool:
    x, y = float(point[0]), float(point[1])
    polygon = normalize_polygon_points(polygon_points)
    inside = False

    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if point_on_segment((x, y), first, second):
            return True

        xi, yi = first
        xj, yj = second
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside

    return inside


def point_on_segment(point: Point2D, first: Point2D, second: Point2D) -> bool:
    px, py = point
    x1, y1 = first
    x2, y2 = second
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if not math.isclose(cross, 0.0, abs_tol=1e-9):
        return False
    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= 1e-9


def distance_2d(first: Point2D, second: Point2D) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def normalize_positive(angle: float) -> float:
    angle = math.fmod(float(angle), 2.0 * math.pi)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle
