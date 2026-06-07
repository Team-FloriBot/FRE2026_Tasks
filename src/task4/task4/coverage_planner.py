#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, List, Sequence, Tuple

import fields2cover as f2c


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class CoveragePlannerConfig:
    operating_width: float = 1.0
    robot_width: float = 0.42
    headland_width: float = 0.5
    turn_radius: float = 0.37


def validate_polygon_coords(coords: Iterable[float]) -> Tuple[bool, str]:
    if (
        isinstance(coords, (str, bytes))
        or not hasattr(coords, "__iter__")
        or not hasattr(coords, "__len__")
    ):
        return False, "polygon_coords muss eine Liste aus Zahlen sein."

    coords = list(coords)

    if len(coords) % 2 != 0:
        return False, "polygon_coords muss eine gerade Anzahl an Elementen besitzen."

    if len(coords) < 6:
        return False, "polygon_coords muss mindestens drei Punkte enthalten."

    for value in coords:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False, "polygon_coords darf nur numerische Werte enthalten."
        if not math.isfinite(float(value)):
            return False, "polygon_coords darf keine NaN- oder Infinity-Werte enthalten."

    return validate_polygon_points(coords_to_points(coords))


def validate_polygon_points(points: Sequence[Point2D]) -> Tuple[bool, str]:
    points = normalize_polygon_points(points)

    if len(points) < 3:
        return False, "polygon_coords muss mindestens drei unterschiedliche Polygonpunkte enthalten."

    if len(set(points)) < 3:
        return False, "polygon_coords muss mindestens drei unterschiedliche Polygonpunkte enthalten."

    if math.isclose(signed_area_twice(points), 0.0, abs_tol=1e-9):
        return False, "polygon_coords darf kein degeneriertes Polygon bilden."

    return True, ""


def normalize_polygon_points(points: Sequence[Point2D]) -> List[Point2D]:
    normalized = [(float(point[0]), float(point[1])) for point in points]
    if len(normalized) > 1 and same_point(normalized[0], normalized[-1]):
        normalized = normalized[:-1]
    return normalized


def coords_to_points(coords: Sequence[float]) -> List[Point2D]:
    return [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]


def points_to_coords(points: Sequence[Point2D]) -> List[float]:
    return [coord for point in points for coord in point]


def same_point(first: Point2D, second: Point2D, abs_tol: float = 1e-9) -> bool:
    return (
        math.isclose(first[0], second[0], abs_tol=abs_tol)
        and math.isclose(first[1], second[1], abs_tol=abs_tol)
    )


def signed_area_twice(points: Sequence[Point2D]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area


def plan_coverage_path(
    polygon_points: Sequence[Point2D],
    start_xy: Point2D,
    config: CoveragePlannerConfig,
) -> List[Waypoint]:
    polygon_points = normalize_polygon_points(polygon_points)
    success, reason = validate_polygon_points(polygon_points)
    if not success:
        raise ValueError(reason)

    ring = f2c.LinearRing()
    for x, y in polygon_points:
        ring.addPoint(float(x), float(y))
    ring.addPoint(float(polygon_points[0][0]), float(polygon_points[0][1]))

    cell = f2c.Cell(ring)
    cells = f2c.Cells(cell)

    robot = f2c.Robot(float(config.robot_width), float(config.operating_width))
    robot.setMinTurningRadius(float(config.turn_radius))

    hl_gen = f2c.HG_Const_gen()
    route_hl_width = max(0.0, float(config.headland_width) / 2.0)
    coverage_hl_width = max(0.0, float(config.headland_width))
    mid_hl = hl_gen.generateHeadlands(cells, route_hl_width)
    no_hl = hl_gen.generateHeadlands(cells, coverage_hl_width)

    swath_generator = f2c.SG_BruteForce()
    swath_objective = f2c.OBJ_NSwath()
    swaths = swath_generator.generateBestSwaths(
        swath_objective,
        robot.getCovWidth(),
        no_hl.getGeometry(0),
    )

    route_planner = f2c.RP_RoutePlannerBase()
    route_planner.setStartAndEndPoint(f2c.Point(float(start_xy[0]), float(start_xy[1])))

    swaths_by_cells = f2c.SwathsByCells()
    try:
        swaths_by_cells.append(swaths)
    except AttributeError:
        swaths_by_cells.push_back(swaths)

    route = route_planner.genRoute(mid_hl, swaths_by_cells)

    path_planner = f2c.PP_PathPlanning()
    dubins = f2c.PP_DubinsCurves()
    f2c_path = path_planner.planPath(robot, route, dubins)

    return waypoints_from_f2c_path(f2c_path)


def waypoints_from_f2c_path(f2c_path) -> List[Waypoint]:
    states = []
    if hasattr(f2c_path, "states"):
        states = f2c_path.states
    elif hasattr(f2c_path, "getStates"):
        states = f2c_path.getStates()
    else:
        states = f2c_path

    waypoints = []
    for state in states:
        waypoints.append(
            Waypoint(
                x=float(state.point.getX()),
                y=float(state.point.getY()),
                yaw=float(state.angle),
            )
        )
    return waypoints
