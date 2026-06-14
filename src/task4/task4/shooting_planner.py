#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from task4.coverage_planner import (
    Point2D,
    Waypoint,
    normalize_polygon_points,
    signed_area_twice,
)


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
    headland_width: float = 0.5


@dataclass(frozen=True)
class ShootingPose:
    x: float
    y: float
    yaw: float
    target_ids: List[int] = field(default_factory=list)


def plan_shooting_poses(
    targets: Sequence[ShotTarget],
    polygon_points: Sequence[Point2D],
    coverage_path: Sequence[Waypoint],
    start_xy: Point2D,
    config: ShootingPlannerConfig,
) -> Tuple[List[ShootingPose], List[ShotTarget]]:
    del coverage_path, start_xy

    polygon = normalize_polygon_points(polygon_points)
    unique_targets = _unique_targets_inside_headland_area(
        targets,
        polygon,
        float(config.headland_width),
    )
    if not unique_targets:
        return [], []

    shooting_pose = _longest_edge_shooting_pose(polygon, unique_targets, config)
    covered_ids = set(shooting_pose.target_ids)
    uncovered = [
        target for target in unique_targets
        if int(target.target_id) not in covered_ids
    ]

    if not shooting_pose.target_ids:
        return [], uncovered

    return [shooting_pose], uncovered


def _unique_targets_inside_headland_area(
    targets: Sequence[ShotTarget],
    polygon: Sequence[Point2D],
    headland_width: float,
) -> List[ShotTarget]:
    min_boundary_distance = max(0.0, float(headland_width))
    by_id = {}
    for target in targets:
        point = (target.x, target.y)
        if (
            point_in_polygon(point, polygon)
            and distance_to_polygon_boundary(point, polygon) >= min_boundary_distance
        ):
            by_id[int(target.target_id)] = target
    return list(by_id.values())


def _longest_edge_shooting_pose(
    polygon: Sequence[Point2D],
    targets: Sequence[ShotTarget],
    config: ShootingPlannerConfig,
) -> ShootingPose:
    start, end = _longest_edge_with_interior_left(polygon)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("Laengste Polygonkante ist degeneriert.")

    ux = dx / length
    uy = dy / length
    left_normal = (-uy, ux)
    midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    offset = max(0.0, float(config.headland_width)) * 0.5
    shooter_xy = _inset_point_inside_polygon(midpoint, left_normal, offset, polygon)
    yaw = normalize_angle(math.atan2(uy, ux))

    covered_ids = [
        int(target.target_id)
        for target in targets
        if target_reachable(shooter_xy, yaw, target, config)
    ]

    return ShootingPose(
        x=shooter_xy[0],
        y=shooter_xy[1],
        yaw=yaw,
        target_ids=sorted(covered_ids),
    )


def _longest_edge_with_interior_left(polygon: Sequence[Point2D]) -> Tuple[Point2D, Point2D]:
    if len(polygon) < 2:
        raise ValueError("Polygon enthaelt zu wenige Punkte.")

    best_start = polygon[0]
    best_end = polygon[1]
    best_length = -1.0

    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        length = distance_2d(start, end)
        if length > best_length:
            best_start = start
            best_end = end
            best_length = length

    if signed_area_twice(polygon) < 0.0:
        return best_end, best_start

    return best_start, best_end


def _inset_point_inside_polygon(
    point: Point2D,
    direction: Point2D,
    offset: float,
    polygon: Sequence[Point2D],
) -> Point2D:
    for ratio in (1.0, 0.75, 0.5, 0.25, 0.0):
        candidate = (
            float(point[0]) + float(direction[0]) * offset * ratio,
            float(point[1]) + float(direction[1]) * offset * ratio,
        )
        if point_in_polygon(candidate, polygon):
            return candidate

    return (float(point[0]), float(point[1]))


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


def distance_to_polygon_boundary(point: Point2D, polygon_points: Sequence[Point2D]) -> float:
    polygon = normalize_polygon_points(polygon_points)
    if len(polygon) < 2:
        return 0.0

    return min(
        distance_to_segment(point, first, polygon[(index + 1) % len(polygon)])
        for index, first in enumerate(polygon)
    )


def distance_to_segment(point: Point2D, first: Point2D, second: Point2D) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(first[0]), float(first[1])
    x2, y2 = float(second[0]), float(second[1])
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)


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
