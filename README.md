# FRE2026
Dieses Repository beinhaltet die Implementierung der Aufgaben für das Field Robot Event 2026.

## Repository klonen und bauen
um das Repo zu bauen, folgendes auführen:

```
git clone https://github.com/Team-FloriBot/FRE2026.git
```

```
cd FRE2026
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

## Starte Base
Für das FRE2026 werden FloriBot1.0 und FloriBOt 4.0 berücksichtigt. Somit kann einer der beiden base-nodes gestaret werden, je nach Verwendung.

... implementierung folgt

