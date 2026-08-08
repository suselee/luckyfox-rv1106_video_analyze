from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any


MessageCallback = Callable[[str, bytes], Awaitable[None] | None]
StateCallback = Callable[[dict[str, Any]], None]


def encode_remaining_length(length: int) -> bytes:
    if length < 0 or length > 268_435_455:
        raise ValueError("invalid MQTT remaining length")
    encoded = bytearray()
    while True:
        digit = length % 128
        length //= 128
        if length:
            digit |= 0x80
        encoded.append(digit)
        if not length:
            return bytes(encoded)


def encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    if len(data) > 65_535:
        raise ValueError("MQTT string is too long")
    return len(data).to_bytes(2, "big") + data


def connect_packet(
    *, client_id: str, username: str, password: str, keepalive_seconds: int
) -> bytes:
    if password and not username:
        raise ValueError("MQTT password requires a username")
    flags = 0x02  # clean session
    if username:
        flags |= 0x80
    if password:
        flags |= 0x40
    body = bytearray()
    body += encode_string("MQTT")
    body += bytes((4, flags))
    body += max(1, min(65_535, keepalive_seconds)).to_bytes(2, "big")
    body += encode_string(client_id)
    if username:
        body += encode_string(username)
    if password:
        body += encode_string(password)
    return b"\x10" + encode_remaining_length(len(body)) + bytes(body)


def subscribe_packet(*, packet_id: int, topic: str, qos: int = 1) -> bytes:
    body = packet_id.to_bytes(2, "big") + encode_string(topic) + bytes((qos & 1,))
    return b"\x82" + encode_remaining_length(len(body)) + body


def subscribe_topics_packet(
    *, packet_id: int, topics: Sequence[str], qos: int = 1
) -> bytes:
    if not topics:
        raise ValueError("at least one MQTT topic is required")
    body = bytearray(packet_id.to_bytes(2, "big"))
    for topic in topics:
        body += encode_string(topic)
        body += bytes((qos & 1,))
    return b"\x82" + encode_remaining_length(len(body)) + bytes(body)


class MQTTSubscriber:
    """Small MQTT 3.1.1 QoS 1 subscriber built only on asyncio streams."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: str,
        topic: str | Sequence[str],
        username: str = "",
        password: str = "",
        keepalive_seconds: int = 30,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.topics = (topic,) if isinstance(topic, str) else tuple(topic)
        if not self.topics:
            raise ValueError("at least one MQTT topic is required")
        self.topic = self.topics[0]
        self.username = username
        self.password = password
        self.keepalive_seconds = max(5, keepalive_seconds)

    @staticmethod
    async def _read_packet(
        reader: asyncio.StreamReader,
    ) -> tuple[int, bytes]:
        first = (await reader.readexactly(1))[0]
        multiplier = 1
        remaining = 0
        for _ in range(4):
            digit = (await reader.readexactly(1))[0]
            remaining += (digit & 0x7F) * multiplier
            if not digit & 0x80:
                break
            multiplier *= 128
        else:
            raise ValueError("malformed MQTT remaining length")
        return first, await reader.readexactly(remaining)

    @staticmethod
    def _publish_payload(first: int, body: bytes) -> tuple[str, bytes, int | None]:
        if len(body) < 2:
            raise ValueError("malformed MQTT publish packet")
        topic_length = int.from_bytes(body[:2], "big")
        cursor = 2
        topic_end = cursor + topic_length
        if topic_end > len(body):
            raise ValueError("malformed MQTT publish topic")
        topic = body[cursor:topic_end].decode("utf-8")
        cursor = topic_end
        qos = (first >> 1) & 0x03
        packet_id = None
        if qos:
            if cursor + 2 > len(body):
                raise ValueError("missing MQTT publish packet id")
            packet_id = int.from_bytes(body[cursor : cursor + 2], "big")
            cursor += 2
        return topic, body[cursor:], packet_id

    async def _session(
        self,
        callback: MessageCallback,
        state_callback: StateCallback,
        stop_event: asyncio.Event,
    ) -> None:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            writer.write(
                connect_packet(
                    client_id=self.client_id,
                    username=self.username,
                    password=self.password,
                    keepalive_seconds=self.keepalive_seconds,
                )
            )
            await writer.drain()
            first, body = await asyncio.wait_for(
                self._read_packet(reader), timeout=10
            )
            if first != 0x20 or len(body) != 2 or body[1] != 0:
                code = body[1] if len(body) >= 2 else "invalid"
                raise ConnectionError(f"MQTT CONNACK rejected: {code}")

            packet_id = 1
            writer.write(
                subscribe_topics_packet(packet_id=packet_id, topics=self.topics, qos=1)
            )
            await writer.drain()
            first, body = await asyncio.wait_for(
                self._read_packet(reader), timeout=10
            )
            if first != 0x90 or len(body) < 2 + len(self.topics) or body[:2] != packet_id.to_bytes(2, "big"):
                raise ConnectionError("MQTT SUBACK missing or invalid")
            if any(code == 0x80 for code in body[2 : 2 + len(self.topics)]):
                raise ConnectionError("MQTT subscription rejected")

            state_callback({"status": "connected", "host": self.host, "port": self.port})
            while not stop_event.is_set():
                try:
                    first, body = await asyncio.wait_for(
                        self._read_packet(reader), timeout=self.keepalive_seconds
                    )
                except asyncio.TimeoutError:
                    writer.write(b"\xc0\x00")
                    await writer.drain()
                    first, body = await asyncio.wait_for(
                        self._read_packet(reader), timeout=self.keepalive_seconds
                    )
                packet_type = first >> 4
                if packet_type == 3:
                    topic, payload, publish_id = self._publish_payload(first, body)
                    outcome = callback(topic, payload)
                    if inspect.isawaitable(outcome):
                        await outcome
                    # QoS 1 is acknowledged only after the callback has
                    # durably accepted the message. If persistence raises, the
                    # connection closes without PUBACK and the broker retries.
                    if publish_id is not None:
                        writer.write(b"\x40\x02" + publish_id.to_bytes(2, "big"))
                        await writer.drain()
                elif packet_type in {9, 13}:
                    continue
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def run(
        self,
        callback: MessageCallback,
        state_callback: StateCallback,
        stop_event: asyncio.Event,
    ) -> None:
        delay = 1
        while not stop_event.is_set():
            try:
                state_callback(
                    {"status": "connecting", "host": self.host, "port": self.port}
                )
                await self._session(callback, state_callback, stop_event)
                delay = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state_callback(
                    {
                        "status": "disconnected",
                        "host": self.host,
                        "port": self.port,
                        "message": str(exc),
                        "retry_seconds": delay,
                    }
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(60, delay * 2)


def decode_json_payload(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("MQTT payload must be a JSON object")
    return value
