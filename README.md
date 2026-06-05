# FRE2026
Dieses Repository beinhaltet die Implementierung der Aufgaben für das Field Robot Event 2026.

## Repository klonen und bauen
um das Repo zu bauen, folgendes auführen:

```
git clone https://github.com/Team-FloriBot/FRE2026_Tasks.git
```

```
cd FRE2026_Tasks
```

```
git submodule init
```

```
git submodule update
```


```
rosdep install -i --from-path src --rosdistro jazzy -y
```

```
colcon build
```

```
source /opt/ros/jazzy/setup.bash
```

```
source install/local_setup.bash
```

## Package Maize Navigation
Enthält die Navigationslogik für den FloriBot im Rahmen des **Field Robot Event 2026**. Der Fokus liegt auf der autonomen Navigation durch Maisreihen basierend auf einem vordefinierten Muster (Pattern).
```
ros2 launch maize_navigation maize_navigation.launch.py
``` 
Die Konfiguration erfolgt über die config/params.yaml. Hier können das Fahrmuster sowie wietere Paramter angepasst werden.

Zum Starten der Maisnavigation folgenden Service in der Kommandozeile aufrufen:
Ohne Object Detection kann `model_path` leer bleiben:
```
ros2 service call /start_navigation maize_navigation_interfaces/srv/StartNavigation "{pattern: '3L 6R 5R', carefulness: 'high', model_path: ''}"
```
Mit Object Detection:
```
ros2 service call /start_navigation maize_navigation_interfaces/srv/StartNavigation "{pattern: '3L 6R 5R', carefulness: 'high', model_path: '/path/to/model.pt'}"
```
`carefulness` waehlt das Fahrprofil. Der erste Teil bestimmt die generellen Fahrparameter:

| Wert | Bedeutung |
| --- | --- |
| `high` | vorsichtigstes Profil |
| `medium` | mittleres Profil |
| `low` | schnellstes Profil |

Der zweite Teil bestimmt, wie stark Laser- und Karten-Zielpunkt gemischt werden:

| Wert | Bedeutung |
| --- | --- |
| `laser` | faehrt hauptsaechlich nach Laser |
| `mix` | mischt Laser und Karte |
| `map` | faehrt hauptsaechlich nach Karte |

Damit gibt es fuer den Startservice diese eindeutigen Optionen:

| `carefulness` | Fahrparameter | Zielgewichtung |
| --- | --- | --- |
| `high_laser` | high | laser |
| `high_mix` | high | mix |
| `high_map` | high | map |
| `medium_laser` | medium | laser |
| `medium_mix` | medium | mix |
| `medium_map` | medium | map |
| `low_laser` | low | laser |
| `low_mix` | low | mix |
| `low_map` | low | map |

Die alten Werte funktionieren weiterhin als Kurzformen: `high` entspricht `high_map`, `medium` entspricht `medium_mix` und `low` entspricht `low_laser`.

Weitere Navigationsservices:
```
ros2 service call /pause_navigation std_srvs/srv/Trigger {}
ros2 service call /resume_navigation std_srvs/srv/Trigger {}
ros2 service call /stop_navigation std_srvs/srv/Trigger {}
ros2 service call /reset_navigation std_srvs/srv/Trigger {}
```


## Task 4

Zum Starten der Task4 an den Startpunk am Feldrand fahren und folgenden Service in der Kommandozeile aufrufen:
```
ros2 service call /trigger_coverage_planning std_srvs/srv/Trigger "{}"
``` 

Eckkoordinaten setzen gemessen vom Startpunkt des Roboters:
```
ros2 param set /coverage_planner polygon_coords '[0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0]'
```

Unterbrechung (Stop)
```
ros2 service call /pure_pursuit_node/set_active std_srvs/srv/SetBool "{data: false}"
```

Starten (nach Stop)
```
ros2 service call /pure_pursuit_node/set_active std_srvs/srv/SetBool "{data: true}"
```
Da wir mehrere Laser am Roboter verbaut haben, müssen wir zum Testen zwischen den Lasern umschalten können. Aktuell kann man zwischen 3 Laserscantopics für den front_laser umschalten:

```
ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'rs_crop_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'rs_nonground_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'sick_front'}"
```
Profile: rs_crop_scan (Robosense Scan nur Pflanzen), rs_nonground_scan (Robosense Scan alles außer Boden), sick_front (2D Laser oder Simulation)
