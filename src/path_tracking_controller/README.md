# Pure-Pursuit-Path-Tracking-Controller

Dieses Package implementiert einen minimalistischen Pure-Pursuit-Controller für Outdoor-Feldroboter mit Knicklenkung.

## Features

- Pure-Pursuit-Algorithmus für Track-Tracking
- Adaptive Lookahead-Distanz basierend auf Geschwindigkeit
- Krümmungsabhängige Geschwindigkeitsregelung für Kurven
- Zielerkennung mit Stop-Funktion
- RViz-Debug-Marker für nearest und lookahead point
- Minimale Abhängigkeiten (kein Nav2-Stack)

## Architektur

```
path_tracking_controller/
├── pure_pursuit_node.cpp    # Haupt-Implementierung
├── params.yaml              # Konfigurationsparameter
├── CMakeLists.txt           # Build-Konfiguration
├── package.xml              # Package-Metadaten
└── launch/
    └── path_tracking_controller.launch.py
```

## ROS2-Knoten: pure_pursuit_node

### Subscriber

- `/path` (nav_msgs/msg/Path): Pfad im map-frame
- `/odom` (nav_msgs/msg/Odometry): Odometrie für Fahrzeugpose und Geschwindigkeit

### Publisher

- `/cmd_vel` (geometry_msgs/msg/Twist): Geschwindigkeitskommandos
- `~/debug_marker` (visualization_msgs/msg/Marker): RViz-Debug-Marker

### TF-Frames

- `map` → `odom` → `base_link`

## Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|-------------|
| `path_topic` | string | `/path` | Topic für den Pfad |
| `odom_topic` | string | `/odom` | Topic für Odometrie |
| `cmd_vel_topic` | string | `/cmd_vel` | Topic für Velocity-Kommandos |
| `map_frame` | string | `map` | Map-Frame für TF |
| `odom_frame` | string | `odom` | Odom-Frame für TF |
| `base_link_frame` | string | `base_link` | Base-Link-Frame für TF |
| `lookahead_min` | double | `0.5` | Minimale Lookahead-Distanz [m] |
| `lookahead_gain` | double | `1.0` | Gain für adaptive Lookahead-Berechnung |
| `max_speed` | double | `2.0` | Maximale Geschwindigkeit [m/s] |
| `min_speed` | double | `0.3` | Minimale Geschwindigkeit [m/s] |
| `curvature_speed_gain` | double | `2.0` | Gain für krümmungsabhängige Geschwindigkeitsreduktion |
| `goal_tolerance` | double | `0.15` | Toleranz für Zielerkennung [m] |
| `control_rate` | double | `30` | Regelrate [Hz] |

## Algorithmus

### 1. Nearest Point
Finde den nächsten Punkt auf dem Pfad relativ zur Fahrzeugposition.

### 2. Lookahead Point
Finde einen Punkt mit einer bestimmten Lookahead-Distanz entlang des Pfades.

### 3. Adaptive Lookahead-Distanz
```
Ld = L_min + k_v * v
```

### 4. Krümmung berechnen
```
κ = 2y / Ld²
```

### 5. Twist berechnen
```
ω = v * κ
```

### 6. Adaptive Geschwindigkeit
```
v = v_max * e^(-k * |κ|)
```

## Installation und Build

```bash
cd FRE2026_Tasks
rosdep install -i --from-path src --rosdistro jazzy -y
colcon build
```

## Start

```bash
source install/local_setup.bash
ros2 launch path_tracking_controller path_tracking_controller.launch.py
```

## Verwendung

1. Pfad über `/path` Topic publizieren (nav_msgs/Path im map-frame)
2. Odometrie über `/odom` Topic publizieren
3. Geschwindigkeitskommandos empfangen auf `/cmd_vel`

## Hinweise

- Dies ist ein minimalistischer Controller ohne Hindernisvermeidung
- Für Feldroboter mit Knicklenkung optimiert
- Kurven schneiden ist akzeptabel
- Kein vollständiger Nav2-Stack erforderlich
