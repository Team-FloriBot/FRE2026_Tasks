#!/usr/bin/env python3

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


@dataclass(frozen=True)
class AudioRequest:
    """Normalized audio request.

    `text` is spoken as-is. If `text` is empty, the phrase is generated from
    `label` and optional abstract navigation hints.
    """

    label: str = ""
    text: Optional[str] = None
    side: Optional[str] = None
    count: Optional[int] = None
    option: Optional[str] = None
    row: Optional[int] = None
    distance_m: Optional[float] = None


class AudioFeedbackNode(Node):
    """Text-to-speech feedback from direct text or abstract object events."""

    def __init__(self) -> None:
        super().__init__("audio_feedback_node")

        # Backwards-compatible parameter name. The topic now accepts generic
        # audio requests, not only classification results.
        self.declare_parameter("classification_topic", "/classification_result")
        self.declare_parameter("spoken_topic", "/audio_feedback/last_phrase")
        self.declare_parameter("enabled", True)
        self.declare_parameter("min_repeat_interval_sec", 1.5)
        self.declare_parameter("speech_rate_wpm", 150)
        self.declare_parameter("speech_volume", 180)
        self.declare_parameter("tts_command", "")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("language", "en")
        self.declare_parameter("speak_healthy", False)

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
        self.speak_healthy = bool(self.get_parameter("speak_healthy").value)
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
            self._on_audio_request,
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
            f"Listening for audio requests on '{classification_topic}'"
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

    def _on_audio_request(self, msg: String) -> None:
        request = self._parse_request(msg.data)
        if request is None:
            self.get_logger().warn(f"Ignoring unsupported audio payload: {msg.data}")
            return

        phrase = self._phrase_for_request(request)
        if phrase is None:
            return

        now = time.monotonic()
        repeat_key = self._repeat_key(request, phrase)
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

    def _parse_request(self, payload: str) -> Optional[AudioRequest]:
        raw = payload.strip()
        if not raw:
            return None

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            direct_text = self._first_string(
                data,
                "text",
                "message",
                "phrase",
                "say",
                "speech",
            )
            if direct_text:
                return AudioRequest(text=direct_text)

            label = self._first_string(
                data,
                "object",
                "object_label",
                "label",
                "class",
                "classification",
                "result",
            )
            option = self._first_string(data, "option", "position", "hint", "direction")
            side = self._first_string(data, "side")
            count = data.get("count", data.get("number"))
            row = data.get("row")
            distance = data.get("distance_m", data.get("distance"))

            option_side, option_count = self._parse_option(option)
            return AudioRequest(
                label=self._normalize_label(label),
                side=self._normalize_side(side) or option_side,
                count=self._parse_int(count) or option_count,
                option=option,
                row=self._parse_int(row),
                distance_m=self._parse_float(distance),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Plain string mode:
        # - known legacy labels are converted to phrases
        # - everything else is interpreted as the exact phrase to speak
        legacy_request = self._parse_legacy_label(raw)
        if legacy_request is not None:
            return legacy_request
        return AudioRequest(text=raw)

    def _first_string(self, data: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = data.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return ""

    def _parse_legacy_label(self, raw: str) -> Optional[AudioRequest]:
        lowered = raw.lower().strip()
        label = self._normalize_known_label(lowered)
        if not label:
            return None

        side = None
        if "left" in lowered or "links" in lowered:
            side = "left"
        elif "right" in lowered or "rechts" in lowered:
            side = "right"

        option_side, option_count = self._parse_option(lowered)
        return AudioRequest(
            label=label,
            side=side or option_side,
            count=option_count,
            option=lowered,
        )

    def _normalize_label(self, text: str) -> str:
        text = text.lower().strip()
        return self._normalize_known_label(text) or text

    def _normalize_known_label(self, text: str) -> str:
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
        return ""

    def _normalize_side(self, side: Optional[str]) -> Optional[str]:
        if side is None:
            return None
        side = side.lower().strip()
        if side in ("left", "links", "l"):
            return "left"
        if side in ("right", "rechts", "r"):
            return "right"
        return None

    def _parse_option(self, option: Optional[str]) -> tuple[Optional[str], Optional[int]]:
        if option is None:
            return None, None

        normalized = option.lower().strip().replace("-", "_").replace(" ", "_")
        side = None
        if "left" in normalized or "links" in normalized:
            side = "left"
        elif "right" in normalized or "rechts" in normalized:
            side = "right"

        count = None
        match = re.search(r"(?<!\d)([12])(?!\d)", normalized)
        if match:
            count = int(match.group(1))
        elif any(token in normalized for token in ("one", "single", "eins", "ein_")):
            count = 1
        elif any(token in normalized for token in ("two", "double", "zwei")):
            count = 2

        return side, count

    def _parse_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _phrase_for_request(self, request: AudioRequest) -> Optional[str]:
        if request.text:
            return request.text

        if not request.label:
            return None

        if request.label == "healthy" and not self.speak_healthy:
            return None

        if request.label == "diseased":
            phrase = "diseased plant detected"
        elif request.label == "bee":
            phrase = "bee - good"
        elif request.label == "pest":
            phrase = "pest detected"
        elif request.label == "butterfly":
            phrase = "neutral"
        elif request.label == "healthy":
            phrase = "healthy plant"
        else:
            phrase = request.label.replace("_", " ")

        parts = [phrase]
        if request.count is not None and request.side in ("left", "right"):
            parts.append(f"{self._count_word(request.count)} {request.side}")
        elif request.side in ("left", "right"):
            parts.append(f"on the {request.side}")
        elif request.count is not None:
            parts.append(self._count_word(request.count))

        if request.row is not None:
            parts.append(f"in row {request.row}")
        return " ".join(parts)

    def _count_word(self, count: int) -> str:
        return {1: "one", 2: "two"}.get(count, str(count))

    def _repeat_key(self, request: AudioRequest, phrase: str) -> str:
        if request.text:
            return phrase
        if request.row is not None and request.distance_m is not None:
            return f"{request.label}:{request.row}:{round(request.distance_m, 1)}"
        return f"{request.label}:{request.side or ''}:{request.count or ''}"

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
