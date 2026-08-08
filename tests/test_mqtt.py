import asyncio
import json

import pytest

from nas_video_summarizer.mqtt import (
    MQTTSubscriber,
    connect_packet,
    decode_json_payload,
    encode_remaining_length,
    subscribe_packet,
    subscribe_topics_packet,
)


def test_remaining_length_encoding():
    assert encode_remaining_length(0) == b"\x00"
    assert encode_remaining_length(127) == b"\x7f"
    assert encode_remaining_length(128) == b"\x80\x01"
    assert encode_remaining_length(16_384) == b"\x80\x80\x01"


def test_connect_and_subscribe_packets_include_credentials_and_topic():
    connect = connect_packet(
        client_id="nas-video",
        username="user",
        password="secret",
        keepalive_seconds=30,
    )
    subscribe = subscribe_packet(packet_id=7, topic="homecam/daughter/hit")

    assert connect[0] == 0x10
    assert b"nas-video" in connect
    assert b"user" in connect
    assert b"secret" in connect
    assert subscribe[0] == 0x82
    assert b"homecam/daughter/hit" in subscribe


def test_multi_topic_subscribe_packet_contains_all_topics():
    packet = subscribe_topics_packet(
        packet_id=3,
        topics=("homecam/daughter/hit", "homecam/daughter/status"),
    )

    assert packet[0] == 0x82
    assert b"homecam/daughter/hit" in packet
    assert b"homecam/daughter/status" in packet


def test_publish_packet_parser_handles_qos1():
    topic = b"homecam/daughter/hit"
    payload = b'{"ts":1,"score":0.5}'
    body = len(topic).to_bytes(2, "big") + topic + b"\x00\x2a" + payload

    parsed_topic, parsed_payload, packet_id = MQTTSubscriber._publish_payload(
        0x32, body
    )

    assert parsed_topic == "homecam/daughter/hit"
    assert parsed_payload == payload
    assert packet_id == 42


def test_decode_json_payload_requires_object():
    assert decode_json_payload(json.dumps({"ts": 1}).encode()) == {"ts": 1}
    with pytest.raises(ValueError):
        decode_json_payload(b"[]")


def test_run_reconnects_after_session_failure():
    subscriber = MQTTSubscriber(
        host="broker", port=1883, client_id="test", topic="topic"
    )
    attempts = 0
    stop_event = asyncio.Event()
    states = []

    async def fake_session(callback, state_callback, supplied_stop_event):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            supplied_stop_event.set()
            return
        raise ConnectionError("offline")

    subscriber._session = fake_session

    asyncio.run(subscriber.run(lambda topic, payload: None, states.append, stop_event))

    assert attempts == 2
    assert any(state["status"] == "disconnected" for state in states)


def test_subscriber_connects_subscribes_and_acknowledges_qos1():
    async def scenario():
        received = []
        states = []
        stop_event = asyncio.Event()
        puback_seen = asyncio.Event()
        callback_completed = asyncio.Event()

        async def broker(reader, writer):
            first, _ = await MQTTSubscriber._read_packet(reader)
            assert first == 0x10
            writer.write(b"\x20\x02\x00\x00")
            await writer.drain()
            first, body = await MQTTSubscriber._read_packet(reader)
            assert first == 0x82
            packet_id = body[:2]
            writer.write(b"\x90\x03" + packet_id + b"\x01")
            topic = b"homecam/daughter/hit"
            payload = b'{"ts":1,"score":0.8}'
            publish_body = (
                len(topic).to_bytes(2, "big") + topic + b"\x00\x2a" + payload
            )
            writer.write(
                b"\x32" + encode_remaining_length(len(publish_body)) + publish_body
            )
            await writer.drain()
            first, body = await MQTTSubscriber._read_packet(reader)
            assert callback_completed.is_set()
            assert first == 0x40 and body == b"\x00\x2a"
            puback_seen.set()
            writer.close()
            await writer.wait_closed()

        try:
            server = await asyncio.start_server(broker, "127.0.0.1", 0)
        except OSError as exc:
            pytest.skip(f"loopback sockets unavailable in this test environment: {exc}")
        port = server.sockets[0].getsockname()[1]
        subscriber = MQTTSubscriber(
            host="127.0.0.1",
            port=port,
            client_id="test-client",
            topic="homecam/daughter/hit",
        )

        async def callback(topic, payload):
            received.append((topic, payload))
            callback_completed.set()
            stop_event.set()

        await asyncio.wait_for(
            subscriber.run(callback, states.append, stop_event), timeout=2
        )
        await asyncio.wait_for(puback_seen.wait(), timeout=1)
        server.close()
        await server.wait_closed()
        assert received == [
            ("homecam/daughter/hit", b'{"ts":1,"score":0.8}')
        ]
        assert any(state["status"] == "connected" for state in states)

    asyncio.run(scenario())
