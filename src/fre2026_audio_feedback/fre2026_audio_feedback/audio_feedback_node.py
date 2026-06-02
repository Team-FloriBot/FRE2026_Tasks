#!/usr/bin/env python3

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


@dataclass(frozen=True)
class ClassificationEvent:
    label: str
    side: Optional[str] = None
    row: Optional[int] = None
    distance_m: Optional[float] = None


class AudioFeedbackNode(Node):
    """Text-to-speech feedback for FRE2026 task 2 and task 3 classifications."""

    def __init__(self) -> None:
        super().__init__("audio_feedback_node")

        self.declare_parameter("classification_topic", "/classification_result")
        self.declare_parameter("spoken_topic", "/audio_feedback/last_phrase")
        self.declare_parameter("enabled", True)
        self.declare_parameter("min_repeat_interval_sec", 1.5)
        self.declare_parameter("speech_rate_wpm", 150)
        self.declare_parameter("speech_volume", 180)
        self.declare_parameter("tts_command", "")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("language", "en")

        classification_topic = self.get_parameter("classification_topic").value
        spoken_topic = self.get_parameter("spoken_topic").value
        self.enabled = bool(self.get_parameter("enabled").value)
        self.min_repeat_interval_sec = float(
            self.get_parameter("min_repeat_interval_sec").value
        )
        self.speech_rate_wpm = int(self.get_parameter("speech_rate_wpm").value)
        self.speech_volume = int(self.get_parameter("speech_volume").value)
        self.language = str(self.get_parameter("language").value)
        self.queue_size = int(self.get_parameter("queue_size").value)
        self.tts_command = self._resolve_tts_command(
            str(self.get_parameter("tts_command").value)
        )

        self._last_spoken_by_key: dict[str, float] = {}
        self._speech_queue: queue.Queue[str] = queue.Queue(maxsize=self.queue_size)
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

        self.subscription = self.create_subscription(
            String,
            classification_topic,
            self._on_classification,
            10,
        )
        self.spoken_pub = self.create_publisher(String, spoken_topic, 10)
        self.enable_srv = self.create_service(
            SetBool,
            "~/set_enabled",
            self._set_enabled,
        )

        if self.tts_command is None:
            self.get_logger().warn(
                "No TTS command found. Install espeak-ng or speech-dispatcher. "
                "The node will still publish the selected phrase."
            )
        else:
            self.get_logger().info(f"Using TTS command: {self.tts_command}")

        self.get_logger().info(
            f"Listening for classification results on '{classification_topic}'"
        )

    def destroy_node(self) -> bool:
        self._worker_stop.set()
        self._worker.join(timeout=1.0)
        return super().destroy_node()

    def _resolve_tts_command(self, configured: str) -> Optional[str]:
        if configured:
            return configured
        for cmd in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(cmd):
                return cmd
        return None

    def _set_enabled(self, request: SetBool.Request, response: SetBool.Response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = "audio feedback enabled" if self.enabled else "audio feedback disabled"
        return response

    def _on_classification(self, msg: String) -> None:
        event = self._parse_event(msg.data)
        if event is None:
            self.get_logger().warn(f"Ignoring unsupported classification payload: {msg.data}")
            return

        phrase = self._phrase_for_event(event)
        if phrase is None:
            return

        now = time.monotonic()
        repeat_key = self._repeat_key(event)
        last_spoken = self._last_spoken_by_key.get(repeat_key, 0.0)
        if now - last_spoken < self.min_repeat_interval_sec:
            return
        self._last_spoken_by_key[repeat_key] = now

        self.spoken_pub.publish(String(data=phrase))
        self.get_logger().info(f"Audio feedback: {phrase}")

        if not self.enabled:
            return
        try:
            self._speech_queue.put_nowait(phrase)
        except queue.Full:
            self.get_logger().warn("Speech queue full. Dropping phrase.")

    def _parse_event(self, payload: str) -> Optional[ClassificationEvent]:
        raw = payload.strip()
        if not raw:
            return None

        try:
            data = json.loads(raw)
            label = str(
                data.get("label")
                or data.get("class")
                or data.get("classification")
                or data.get("result")
                or ""
            ).strip().lower()
            side = data.get("side")
            row = data.get("row")
            distance = data.get("distance_m", data.get("distance"))
            return ClassificationEvent(
                label=self._normalize_label(label),
                side=str(side).lower() if side is not None else None,
                row=int(row) if row is not None else None,
                distance_m=float(distance) if distance is not None else None,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        lowered = raw.lower()
        label = self._normalize_label(lowered)
        if not label:
            return None

        side = None
        if "left" in lowered or "links" in lowered:
            side = "left"
        elif "right" in lowered or "rechts" in lowered:
            side = "right"

        return ClassificationEvent(label=label, side=side)

    def _normalize_label(self, text: str) -> str:
        text = text.lower().strip()
        if any(token in text for token in ("diseased", "unhealthy", "krank", "yellow", "brown")):
            return "diseased"
        if any(token in text for token in ("bee", "biene")):
            return "bee"
        if any(token in text for token in ("pest", "aphid", "beetle", "ladybird", "laus", "kaefer", "käfer", "schädling")):
            return "pest"
        if any(token in text for token in ("butterfly", "neutral", "schmetterling")):
            return "butterfly"
        if any(token in text for token in ("healthy", "gesund")):
            return "healthy"
        return text if text in {"diseased", "bee", "pest", "butterfly", "healthy"} else ""

    def _phrase_for_event(self, event: ClassificationEvent) -> Optional[str]:
        if event.label == "healthy":
            return None

        if event.label == "diseased":
            parts = ["diseased plant detected"]
            if event.side in ("left", "right"):
                parts.append(f"on the {event.side}")
            if event.row is not None:
                parts.append(f"in row {event.row}")
            return " ".join(parts)

        if event.label == "bee":
            return "bee - good"
        if event.label == "pest":
            return "pest detected"
        if event.label == "butterfly":
            return "neutral"
        return None

    def _repeat_key(self, event: ClassificationEvent) -> str:
        if event.row is not None and event.distance_m is not None:
            return f"{event.label}:{event.row}:{round(event.distance_m, 1)}"
        return f"{event.label}:{event.side or ''}"

    def _speech_worker(self) -> None:
        while not self._worker_stop.is_set():
            try:
                phrase = self._speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._speak(phrase)
            self._speech_queue.task_done()

    def _speak(self, phrase: str) -> None:
        if self.tts_command is None:
            return

        if self.tts_command in ("espeak", "espeak-ng"):
            cmd = [
                self.tts_command,
                "-v",
                self.language,
                "-s",
                str(self.speech_rate_wpm),
                "-a",
                str(self.speech_volume),
                phrase,
            ]
        elif self.tts_command == "spd-say":
            cmd = [self.tts_command, phrase]
        else:
            cmd = [self.tts_command, phrase]

        try:
            subprocess.run(cmd, check=False, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"TTS command failed: {exc}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = AudioFeedbackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
