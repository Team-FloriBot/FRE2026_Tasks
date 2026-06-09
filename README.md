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
ros2 service call /start_navigation maize_navigation_interfaces/srv/StartNavigation "{pattern: '3L 6R 5R', carefulness: 'high_map', model_path: '', object_row_range: 0, plant_row_count: 0, max_navigation_duration_sec: 0.0}"
```
Mit Object Detection:
```
ros2 service call /start_navigation maize_navigation_interfaces/srv/StartNavigation "{pattern: '3L 6R 5R', carefulness: 'high_map', model_path: '/path/to/model.pt', object_row_range: 1, plant_row_count: 5, starting_lane_number: 1, row_numbers_increase_to: 'left', max_navigation_duration_sec: 0.0}"
```

Alle Felder des Startservices:

| Feld | Typ | Bedeutung | Standard / Deaktivieren |
| --- | --- | --- | --- |
| `pattern` | `string` | Fahrmuster als Schritte aus Anzahl und Richtung, z. B. `3L 6R 5R`. | Pflichtfeld |
| `carefulness` | `string` | Fahrprofil aus Fahrparametern und Zielgewichtung. | leer oder weggelassen = `high` |
| `model_path` | `string` | Pfad zum Object-Detection-Modell. | leer = Object Detection aus |
| `object_row_range` | `int32` | Anzahl Pflanzenreihen, in denen erkannte Objekte für Stopps berücksichtigt werden. Wirkt nur mit gesetztem `model_path`. | `0` = Objektstopps aus; ohne `model_path` immer aus |
| `plant_row_count` | `int32` | Gesamtzahl der Pflanzenreihen, falls bekannt. Wird für die Zuordnung von Objekten zu Reihen verwendet. | `0` = unbekannt |
| `starting_lane_number` | `int32` | Nummer der Startgasse. Startgasse `1` liegt zwischen Pflanzenreihe 1 und 2. | `0` = Wert aus `params.yaml` |
| `row_numbers_increase_to` | `string` | Feldfeste Richtung, in der die Pflanzenreihennummern vom Roboterstart aus groesser werden: `left` oder `right`. | leer = Wert aus `params.yaml` |
| `max_navigation_duration_sec` | `float64` | Maximale Navigationsdauer in Sekunden. | `0.0` = kein Zeitlimit |

Fuer die Objekt-CSV sollten `starting_lane_number`, `row_numbers_increase_to` und `plant_row_count` passend zum Feldaufbau gesetzt werden. Die CSV verwendet damit eine einheitliche Feldreferenz: `row_number` und `distance_from_start_m`, wobei die Distanz immer von der urspruenglichen Startseite des Feldes in die Reihe hinein gemessen wird.

`carefulness` wählt das Fahrprofil. Der erste Teil bestimmt die generellen Fahrparameter:

| Wert | Bedeutung |
| --- | --- |
| `high` | vorsichtigstes Profil |
| `medium` | mittleres Profil |
| `low` | schnellstes Profil |

Der zweite Teil bestimmt, wie stark Laser- und Karten-Zielpunkt gemischt werden:

| Wert | Bedeutung |
| --- | --- |
| `laser` | fährt hauptsächlich nach Laser |
| `mix` | mischt Laser und Karte |
| `map` | fährt hauptsächlich nach Karte |

Damit gibt es für den Startservice diese eindeutigen Optionen:

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

Beispiel mit Zeitlimit und bekannter Pflanzenreihenzahl:
```
ros2 service call /start_navigation maize_navigation_interfaces/srv/StartNavigation "{pattern: '3L 6R 5R', carefulness: 'medium_mix', model_path: '/path/to/model.pt', object_row_range: 2, plant_row_count: 5, starting_lane_number: 1, row_numbers_increase_to: 'left', max_navigation_duration_sec: 45.0}"
```

Weitere Navigationsservices:
```
ros2 service call /pause_navigation std_srvs/srv/Trigger {}
ros2 service call /resume_navigation std_srvs/srv/Trigger {}
ros2 service call /stop_navigation std_srvs/srv/Trigger {}
ros2 service call /reset_navigation std_srvs/srv/Trigger {}
```


## Task 4

Eckkoordinaten setzen gemessen vom Startpunkt des Roboters:
```
ros2 param set /task4_brain polygon_coords '[0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0]'
```

Zum Planen der Task4 an den Startpunkt am Feldrand fahren und folgenden Service in der Kommandozeile aufrufen:
```
ros2 service call /task4/plan_coverage std_srvs/srv/Trigger "{}"
``` 

Zum Starten der geplanten Navigation:
```
ros2 service call /task4/start_navigation std_srvs/srv/Trigger "{}"
```


Unterbrechung (Stop)
```
ros2 service call /task4/stop_navigation std_srvs/srv/Trigger "{}"
```

Reset:
```
ros2 service call /task4/reset std_srvs/srv/Trigger "{}"
```


## Laser Mux
Da wir mehrere Laser am Roboter verbaut haben, müssen wir zum Testen zwischen den Lasern umschalten können. Aktuell kann man zwischen 4 Laserscantopics für den front_laser umschalten:

```
ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'rs_crop_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'rs_nonground_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'sick_front'}"

ros2 service call /front_scan_mux/set_profile fre2026_tasks_interfaces/srv/SetScanProfile "{profile: 'rs_nonground_scan_torsten'}"
```
Profile: rs_crop_scan (Robosense Scan nur Pflanzen), rs_nonground_scan (Robosense Scan alles außer Boden), sick_front (2D Laser oder Simulation)
