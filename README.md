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
```
ros2 service call /start_navigation std_srvs/srv/Trigger {}
``` 

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
ros2 service call /front_scan_mux/set_profile fre2026_task_interfaces/srv/SetScanProfile "{profile: 'rs_crop_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_task_interfaces/srv/SetScanProfile "{profile: 'rs_nonground_scan'}"

ros2 service call /front_scan_mux/set_profile fre2026_task_interfaces/srv/SetScanProfile "{profile: 'sick_front'}"
```
Profile: rs_crop_scan (Robosense Scan nur Pflanzen), rs_nonground_scan (Robosense Scan alles außer Boden), sick_front (2D Laser oder Simulation)