import asyncio
import base64
import json
import os
import unittest
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import MagicMock, patch

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosedError

from voice_agent import (
    AUDIO_RELAY_LOG_INTERVAL,
    FUNCTION_MAP,
    QueuedAudio,
    deepgram_receiver,
    deepgram_sender,
    exotel_handler,
    exotel_receiver,
    handle_function_call_request,
    handle_text_message,
    initialize_deepgram,
    load_config,
    send_pending_enquiries,
)


class FakeWebSocket:
    def __init__(self, incoming: list[str | bytes]) -> None:
        self.incoming = iter(incoming)
        self.sent: list[str | bytes] = []

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return next(self.incoming)
        except StopIteration:
            raise StopAsyncIteration from None

    async def recv(self) -> str | bytes:
        return await self.__anext__()

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)


class WaitingExotelWebSocket(FakeWebSocket):
    def __init__(self) -> None:
        super().__init__(
            ['{"event":"start","start":{"stream_sid":"stream-123"}}']
        )
        self.stop_event = asyncio.Event()
        self.stop_sent = False

    async def __anext__(self) -> str | bytes:
        try:
            return next(self.incoming)
        except StopIteration:
            await self.stop_event.wait()
            if not self.stop_sent:
                self.stop_sent = True
                return '{"event":"stop"}'
            raise StopAsyncIteration from None


class AbruptlyClosedWebSocket:
    def __aiter__(self) -> "AbruptlyClosedWebSocket":
        return self

    async def __anext__(self) -> str:
        raise ConnectionClosedError(None, None)


class VoiceAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_loads_deepgram_settings(self) -> None:
        settings = load_config()
        self.assertEqual(settings["type"], "Settings")
        self.assertEqual(settings["audio"]["input"]["encoding"], "linear16")
        functions = settings["agent"]["think"]["functions"]
        self.assertEqual(functions[0]["name"], "submit_customer_enquiry")
        self.assertNotIn("client_side", functions[0])
        self.assertTrue(functions[0]["defer_until_eot"])
        prompt = settings["agent"]["think"]["prompt"].lower()
        self.assertIn("never mention email", prompt)
        self.assertIn("never use markdown", prompt)
        self.assertIn("read them back", prompt)
        self.assertIn("brief goodbye and stop speaking", prompt)
        self.assertEqual(
            functions[0]["parameters"]["required"],
            [
                "customer_name",
                "customer_phone",
                "reason_for_call",
                "customer_requirement",
                "preferred_date_time",
                "additional_notes",
            ],
        )

    async def test_waits_for_deepgram_setup_before_sending_audio(self) -> None:
        deepgram = FakeWebSocket(
            [
                '{"type":"Welcome"}',
                '{"type":"Warning","description":"test warning"}',
                '{"type":"SettingsApplied"}',
            ]
        )
        settings = {"type": "Settings"}

        await initialize_deepgram(cast(ClientConnection, deepgram), settings)

        self.assertEqual(deepgram.sent, [json.dumps(settings)])

    async def test_exotel_audio_is_decoded_and_queued(self) -> None:
        audio = b"\x01\x02\x03"
        exotel = FakeWebSocket(
            [
                '{"event":"connected"}',
                '{"event":"start","start":{"stream_sid":"stream-123"}}',
                json.dumps(
                    {
                        "event": "media",
                        "media": {"payload": base64.b64encode(audio).decode("ascii")},
                    }
                ),
                '{"event":"media","media":{"payload":"invalid!"}}',
                '{"event":"stop"}',
            ]
        )
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        await exotel_receiver(
            cast(ServerConnection, exotel),
            audio_queue,
            stream_sid_future,
            {},
        )

        self.assertEqual((await audio_queue.get()).payload, audio)
        self.assertEqual(stream_sid_future.result(), "stream-123")

    async def test_exotel_connection_without_close_frame_is_handled_normally(self) -> None:
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        await exotel_receiver(
            cast(ServerConnection, AbruptlyClosedWebSocket()),
            audio_queue,
            stream_sid_future,
            {},
        )

        self.assertFalse(stream_sid_future.done())

    async def test_audio_sender_forwards_raw_bytes_to_deepgram(self) -> None:
        audio = b"\x01\x02\x03"
        deepgram = FakeWebSocket([])
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        for _ in range(AUDIO_RELAY_LOG_INTERVAL):
            await audio_queue.put(QueuedAudio(audio, loop.time()))

        sender = asyncio.create_task(
            deepgram_sender(cast(ClientConnection, deepgram), audio_queue)
        )
        with self.assertLogs("voice_agent", level="INFO") as captured:
            try:
                for _ in range(20):
                    if len(deepgram.sent) == AUDIO_RELAY_LOG_INTERVAL:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(
                    deepgram.sent,
                    [audio] * AUDIO_RELAY_LOG_INTERVAL,
                )
            finally:
                sender.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await sender

        self.assertTrue(
            any("Audio relay timing" in line for line in captured.output)
        )

    async def test_deepgram_barge_in_clears_exotel_and_relays_audio(self) -> None:
        audio = b"\x10\x20\x30"
        deepgram = FakeWebSocket(
            [
                '{"type":"UserStartedSpeaking"}',
                audio,
                '{"type":"ConversationText","role":"user","content":"hello"}',
                '{"type":"AgentAudioDone"}',
            ]
        )
        exotel = FakeWebSocket([])
        stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        stream_sid_future.set_result("stream-123")

        await deepgram_receiver(
            cast(ClientConnection, deepgram),
            cast(ServerConnection, exotel),
            stream_sid_future,
            {},
        )

        self.assertEqual(
            json.loads(cast(str, exotel.sent[0])),
            {"event": "clear", "stream_sid": "stream-123"},
        )
        media = json.loads(cast(str, exotel.sent[1]))
        self.assertEqual(media["event"], "media")
        self.assertEqual(media["stream_sid"], "stream-123")
        self.assertEqual(
            base64.b64decode(media["media"]["payload"]),
            audio,
        )

    async def test_deepgram_latency_report_logs_stage_timings(self) -> None:
        deepgram = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "LatencyReport",
                        "stt_latency": 0.12,
                        "ttt_token_latency": 0.34,
                        "tts_latency": 0.18,
                        "total_latency": 0.64,
                    }
                )
            ]
        )
        exotel = FakeWebSocket([])
        stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        stream_sid_future.set_result("stream-123")

        with self.assertLogs("voice_agent", level="INFO") as captured:
            await deepgram_receiver(
                cast(ClientConnection, deepgram),
                cast(ServerConnection, exotel),
                stream_sid_future,
                {},
            )

        latency_logs = [
            line for line in captured.output if "Deepgram turn latency" in line
        ]
        self.assertEqual(len(latency_logs), 1)
        self.assertIn('"total_latency": 0.64', latency_logs[0])

    async def test_function_call_stages_email_and_replies_to_deepgram(self) -> None:
        arguments = {
            "customer_name": "Alex",
            "customer_phone": "555-0100",
            "reason_for_call": "Service question",
            "customer_requirement": "A repair",
            "preferred_date_time": "Friday morning",
            "additional_notes": "Call back after 10",
        }
        deepgram = FakeWebSocket([])
        pending_enquiries: dict[str, dict[str, str]] = {}
        function_call = {
            "id": "call-123",
            "name": "submit_customer_enquiry",
            "client_side": True,
            "arguments": json.dumps(arguments),
            "thought_signature": "signature-123",
        }

        with patch.dict(
            FUNCTION_MAP,
            {
                "submit_customer_enquiry": lambda enquiry: {
                    "success": True,
                    "message": f"Sent for {enquiry['customer_name']}",
                }
            },
        ):
            await handle_function_call_request(
                {"type": "FunctionCallRequest", "functions": [function_call]},
                cast(ClientConnection, deepgram),
                pending_enquiries,
            )

        self.assertEqual(pending_enquiries, {"call-123": arguments})
        self.assertEqual(len(deepgram.sent), 1)
        response = json.loads(cast(str, deepgram.sent[0]))
        self.assertEqual(response["type"], "FunctionCallResponse")
        self.assertEqual(response["id"], "call-123")
        self.assertEqual(response["name"], "submit_customer_enquiry")
        self.assertEqual(response["thought_signature"], "signature-123")
        self.assertEqual(
            json.loads(response["content"]),
            {"success": True},
        )

    @patch.dict(
        os.environ,
        {
            "BUSINESS_EMAIL": "business@example.com",
            "SMTP_EMAIL": "sender@example.com",
            "SMTP_PASSWORD": "test-password",
        },
        clear=True,
    )
    @patch("voice_agent.email_send.smtplib.SMTP_SSL")
    async def test_tool_request_sends_email_through_configured_function(
        self,
        smtp_ssl: MagicMock,
    ) -> None:
        server = smtp_ssl.return_value.__enter__.return_value
        arguments = {
            "customer_name": "Alex",
            "customer_phone": "555-0100",
            "reason_for_call": "Service question",
            "customer_requirement": "A repair",
            "preferred_date_time": "Friday morning",
            "additional_notes": "Call back after 10",
        }
        deepgram = FakeWebSocket([])
        pending_enquiries: dict[str, dict[str, str]] = {}

        await handle_function_call_request(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {
                        "id": "call-email",
                        "name": "submit_customer_enquiry",
                        "client_side": True,
                        "arguments": json.dumps(arguments),
                    }
                ],
            },
            cast(ClientConnection, deepgram),
            pending_enquiries,
        )

        server.send_message.assert_not_called()
        self.assertEqual(pending_enquiries, {"call-email": arguments})
        await send_pending_enquiries(pending_enquiries)
        server.send_message.assert_called_once()
        response = json.loads(cast(str, deepgram.sent[0]))
        self.assertTrue(json.loads(response["content"])["success"])

    async def test_cancelled_function_call_is_not_emailed(self) -> None:
        arguments = {
            "customer_name": "Alex",
            "customer_phone": "555-0100",
            "reason_for_call": "Service question",
            "customer_requirement": "A repair",
            "preferred_date_time": "Friday morning",
            "additional_notes": "Call back after 10",
        }
        pending_enquiries = {"call-cancelled": arguments}
        deepgram = FakeWebSocket([])

        await handle_text_message(
            {
                "type": "FunctionCallCancelled",
                "functions": [{"id": "call-cancelled", "name": "submit_customer_enquiry"}],
            },
            cast(ServerConnection, FakeWebSocket([])),
            cast(ClientConnection, deepgram),
            "stream-123",
            pending_enquiries,
        )

        self.assertEqual(pending_enquiries, {})
        send_email = MagicMock()
        with patch.dict(
            FUNCTION_MAP,
            {"submit_customer_enquiry": send_email},
        ):
            await send_pending_enquiries(pending_enquiries)
        send_email.assert_not_called()

    async def test_exotel_stop_sends_staged_email_once(self) -> None:
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        stream_sid_future: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        enquiry = {
            "customer_name": "Alex",
            "customer_phone": "555-0100",
            "reason_for_call": "Service question",
            "customer_requirement": "A repair",
            "preferred_date_time": "Friday morning",
            "additional_notes": "Call back after 10",
        }
        pending_enquiries = {"call-stop": enquiry}
        exotel = FakeWebSocket(
            [
                '{"event":"start","start":{"stream_sid":"stream-123"}}',
                '{"event":"stop"}',
            ]
        )
        send_email = MagicMock(return_value={"success": True, "message": "sent"})

        with patch.dict(
            FUNCTION_MAP,
            {"submit_customer_enquiry": send_email},
        ):
            await exotel_receiver(
                cast(ServerConnection, exotel),
                audio_queue,
                stream_sid_future,
                pending_enquiries,
            )
            await send_pending_enquiries(pending_enquiries)

        send_email.assert_called_once_with(enquiry)
        self.assertEqual(pending_enquiries, {})

    async def test_exotel_disconnect_sends_staged_email(self) -> None:
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        stream_sid_future: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        enquiry = {"customer_name": "Alex"}
        pending_enquiries = {"call-disconnect": enquiry}
        send_email = MagicMock(return_value={"success": True, "message": "sent"})

        with patch.dict(
            FUNCTION_MAP,
            {"submit_customer_enquiry": send_email},
        ):
            await exotel_receiver(
                cast(ServerConnection, AbruptlyClosedWebSocket()),
                audio_queue,
                stream_sid_future,
                pending_enquiries,
            )

        send_email.assert_called_once_with(enquiry)
        self.assertEqual(pending_enquiries, {})

    async def test_clean_websocket_close_sends_staged_email(self) -> None:
        audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue()
        stream_sid_future: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        enquiry = {"customer_name": "Alex"}
        pending_enquiries = {"call-clean-close": enquiry}
        send_email = MagicMock(return_value={"success": True, "message": "sent"})

        with patch.dict(
            FUNCTION_MAP,
            {"submit_customer_enquiry": send_email},
        ):
            await exotel_receiver(
                cast(ServerConnection, FakeWebSocket([])),
                audio_queue,
                stream_sid_future,
                pending_enquiries,
            )

        send_email.assert_called_once_with(enquiry)
        self.assertEqual(pending_enquiries, {})

    async def test_handler_reconnects_after_deepgram_transport_failure(self) -> None:
        exotel = WaitingExotelWebSocket()
        connect_calls = 0
        receiver_calls = 0

        @asynccontextmanager
        async def fake_connect(*args: object, **kwargs: object):
            nonlocal connect_calls
            self.assertEqual(args[0], "wss://agent.deepgram.com/v1/agent/converse")
            self.assertIn("additional_headers", kwargs)
            connect_calls += 1
            if connect_calls == 2:
                exotel.stop_event.set()
            yield FakeWebSocket([])

        async def fake_initialize(
            deepgram: ClientConnection,
            settings: dict[str, object],
        ) -> None:
            self.assertIsNotNone(deepgram)
            self.assertEqual(settings["type"], "Settings")
            return None

        async def fake_sender(
            deepgram: ClientConnection,
            audio_queue: asyncio.Queue[QueuedAudio],
        ) -> None:
            self.assertIsNotNone(deepgram)
            self.assertGreater(audio_queue.maxsize, 0)
            await asyncio.Future()

        async def fake_receiver(
            deepgram: ClientConnection,
            exotel_ws: ServerConnection,
            stream_sid: asyncio.Future[str],
            _pending_enquiries: dict[str, dict[str, str]],
        ) -> None:
            nonlocal receiver_calls
            self.assertIsNotNone(deepgram)
            self.assertIsNotNone(exotel_ws)
            self.assertEqual(await stream_sid, "stream-123")
            self.assertIsNotNone(_pending_enquiries)
            receiver_calls += 1
            if receiver_calls == 1:
                raise OSError("simulated TLS transport interruption")
            await exotel.stop_event.wait()

        with (
            patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test-key"}),
            patch("voice_agent.connect", fake_connect),
            patch("voice_agent.initialize_deepgram", fake_initialize),
            patch("voice_agent.deepgram_sender", fake_sender),
            patch("voice_agent.deepgram_receiver", fake_receiver),
            patch("voice_agent.MAX_RECONNECT_DELAY_SECONDS", 0),
        ):
            await asyncio.wait_for(
                exotel_handler(cast(ServerConnection, exotel)),
                timeout=2,
            )

        self.assertEqual(connect_calls, 2)

    async def test_unknown_client_function_returns_error_to_deepgram(self) -> None:
        deepgram = FakeWebSocket([])
        function_call = {
            "id": "call-unknown",
            "name": "unknown_function",
            "client_side": True,
            "arguments": "{}",
        }

        await handle_function_call_request(
            {"type": "FunctionCallRequest", "functions": [function_call]},
            cast(ClientConnection, deepgram),
            {},
        )

        response = json.loads(cast(str, deepgram.sent[0]))
        self.assertEqual(response["id"], "call-unknown")
        self.assertFalse(json.loads(response["content"])["success"])

    async def test_deepgram_receiver_dispatches_function_call_requests(self) -> None:
        arguments = {
            "customer_name": "Alex",
            "customer_phone": "555-0100",
            "reason_for_call": "Service question",
            "customer_requirement": "A repair",
            "preferred_date_time": "Friday morning",
            "additional_notes": "Call back after 10",
        }
        deepgram = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "FunctionCallRequest",
                        "functions": [
                            {
                                "id": "call-456",
                                "name": "submit_customer_enquiry",
                                "client_side": True,
                                "arguments": json.dumps(arguments),
                            }
                        ],
                    }
                )
            ]
        )
        exotel = FakeWebSocket([])
        stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        stream_sid_future.set_result("stream-123")
        pending_enquiries: dict[str, dict[str, str]] = {}

        with patch.dict(
            FUNCTION_MAP,
            {
                "submit_customer_enquiry": lambda enquiry: {
                    "success": True,
                    "message": f"Email sent for {enquiry['customer_name']}",
                },
            },
        ):
            await deepgram_receiver(
                cast(ClientConnection, deepgram),
                cast(ServerConnection, exotel),
                stream_sid_future,
                pending_enquiries,
            )

        response = json.loads(cast(str, deepgram.sent[0]))
        self.assertEqual(response["type"], "FunctionCallResponse")
        self.assertEqual(response["id"], "call-456")
        result = json.loads(response["content"])
        self.assertTrue(result["success"])
        self.assertEqual(result, {"success": True})
        self.assertEqual(pending_enquiries, {"call-456": arguments})


if __name__ == "__main__":
    unittest.main()
