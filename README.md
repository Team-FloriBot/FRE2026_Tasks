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

Zum Starten des Roboters folgenden Service in der Kommandozeile aufrufen:
```
ros2 service call /start_navigation std_srvs/srv/Trigger {}
``` 