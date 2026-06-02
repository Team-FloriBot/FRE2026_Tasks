# fre2026_audio_feedback

ROS 2 Node für gesprochene Rückmeldungen in FRE2026 Task 2 und Task 3.

## Funktion

Der Node abonniert ein Klassifikationstopic und gibt abhängig vom Ergebnis eine gesprochene Meldung über Text-to-Speech aus.

Task 2:

- `diseased`
- `diseased_left`
- `diseased_right`

Task 3:

- `bee`
- `pest`
- `butterfly`

Zusätzlich wird die zuletzt gesprochene Phrase auf einem ROS-Topic veröffentlicht.

## Topic

Default-Input:

```bash
/classification_result  std_msgs/msg/String
```

Default-Output:

```bash
/audio_feedback/last_phrase  std_msgs/msg/String
```

## Unterstützte einfache Payloads

```text
diseased
diseased_left
diseased_right
bee
pest
butterfly
neutral
```

## Unterstützte JSON-Payloads

```json
{"label": "diseased", "side": "left", "row": 2, "distance_m": 8.4}
{"label": "diseased", "side": "right"}
{"label": "bee"}
{"label": "pest"}
{"label": "butterfly"}
```

## Gesprochene Meldungen

| Klassifikation | Ausgabe |
|---|---|
| `diseased` | `diseased plant detected` |
| `diseased_left` | `diseased plant detected on the left` |
| `diseased_right` | `diseased plant detected on the right` |
| `bee` | `bee - good` |
| `pest` | `pest detected` |
| `butterfly` / `neutral` | `neutral` |

## Installation

Text-to-Speech Backend installieren:

```bash
sudo apt install espeak-ng
```

Workspace bauen:

```bash
colcon build --packages-select fre2026_audio_feedback
source install/local_setup.bash
```

## Start

```bash
ros2 launch fre2026_audio_feedback audio_feedback.launch.py
```

## Test

Task 3:

```bash
ros2 topic pub --once /classification_result std_msgs/msg/String "{data: 'bee'}"
ros2 topic pub --once /classification_result std_msgs/msg/String "{data: 'pest'}"
ros2 topic pub --once /classification_result std_msgs/msg/String "{data: 'butterfly'}"
```

Task 2:

```bash
ros2 topic pub --once /classification_result std_msgs/msg/String "{data: 'diseased_left'}"
ros2 topic pub --once /classification_result std_msgs/msg/String "{data: 'diseased_right'}"
```

JSON-Beispiel:

```bash
ros2 topic pub --once /classification_result std_msgs/msg/String \
"{data: '{"label": "diseased", "side": "left", "row": 3, "distance_m": 8.4}'}"
```

## Aktivieren und Deaktivieren

Audioausgabe deaktivieren:

```bash
ros2 service call /audio_feedback_node/set_enabled std_srvs/srv/SetBool "{data: false}"
```

Audioausgabe aktivieren:

```bash
ros2 service call /audio_feedback_node/set_enabled std_srvs/srv/SetBool "{data: true}"
```

## Parameter

Die Parameter liegen in:

```text
config/audio_feedback.yaml
```

Default-Konfiguration:

```yaml
audio_feedback_node:
  ros__parameters:
    classification_topic: /classification_result
    spoken_topic: /audio_feedback/last_phrase
    enabled: true
    min_repeat_interval_sec: 1.5
    speech_rate_wpm: 150
    speech_volume: 180
    language: en
    tts_command: ""
    queue_size: 10
```

### Parameterbeschreibung

| Parameter | Bedeutung |
|---|---|
| `classification_topic` | Topic, auf dem Klassifikationsergebnisse empfangen werden |
| `spoken_topic` | Topic, auf dem die zuletzt gesprochene Phrase veröffentlicht wird |
| `enabled` | Aktiviert oder deaktiviert die Audioausgabe |
| `min_repeat_interval_sec` | Mindestzeit zwischen gleichen Meldungen |
| `speech_rate_wpm` | Sprachgeschwindigkeit in Wörtern pro Minute |
| `speech_volume` | Lautstärke für `espeak` / `espeak-ng` |
| `language` | Sprache/Stimme für Text-to-Speech |
| `tts_command` | Optional festes TTS-Kommando, sonst automatische Suche |
| `queue_size` | Maximale Anzahl wartender Sprachausgaben |

## Integration

Ein Klassifikationsnode muss nur eine `std_msgs/msg/String` Nachricht auf das konfigurierte Topic publizieren.

Python-Beispiel:

```python
from std_msgs.msg import String

msg = String()
msg.data = "bee"
publisher.publish(msg)
```

JSON-Beispiel:

```python
from std_msgs.msg import String
import json

payload = {
    "label": "diseased",
    "side": "left",
    "row": 3,
    "distance_m": 8.4,
}

msg = String()
msg.data = json.dumps(payload)
publisher.publish(msg)
```

## Hinweise

- Gesunde Pflanzen lösen keine Sprachausgabe aus.
- Gleiche Meldungen werden über `min_repeat_interval_sec` entprellt.
- Wenn kein TTS-Backend gefunden wird, publiziert der Node weiterhin die Phrase auf `/audio_feedback/last_phrase`.
- Unterstützte TTS-Kommandos bei automatischer Auswahl: `espeak-ng`, `espeak`, `spd-say`.
