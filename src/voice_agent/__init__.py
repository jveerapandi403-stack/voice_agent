import asyncio
import base64
import binascii
import json
import logging
import os
import smtplib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import load_dotenv
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .email_send import send_customer_enquiry

__all__ = ["main", "send_customer_enquiry"]

logger = logging.getLogger(__name__)
DEEPGRAM_URL = "wss://agent.deepgram.com/v1/agent/converse"
CONFIG_PATH = Path(__file__).with_name("config.json")
QUEUE_SIZE = 100
AUDIO_RELAY_LOG_INTERVAL = 50
HANDSHAKE_TIMEOUT_SECONDS = 15
MAX_RECONNECT_DELAY_SECONDS = 30
FUNCTION_MAP: dict[str, Callable[[Mapping[str, object]], dict[str, object]]] = {
    "submit_customer_enquiry": send_customer_enquiry,
}
ENQUIRY_FIELDS = (
    "customer_name",
    "customer_phone",
    "reason_for_call",
    "customer_requirement",
    "preferred_date_time",
    "additional_notes",
)


@dataclass(frozen=True)
class QueuedAudio:
    payload: bytes
    received_at: float


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict) or config.get("type") != "Settings":
        raise ValueError(f"Invalid Deepgram Settings configuration: {CONFIG_PATH}")
    return config


async def _receive_deepgram_event(deepgram_ws: ClientConnection) -> dict[str, Any]:
    message = await asyncio.wait_for(
        deepgram_ws.recv(),
        timeout=HANDSHAKE_TIMEOUT_SECONDS,
    )
    if not isinstance(message, str):
        raise RuntimeError("Deepgram sent binary data before completing setup")

    try:
        event = json.loads(message)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Deepgram sent invalid JSON during setup") from exc

    if not isinstance(event, dict):
        raise RuntimeError("Deepgram sent a non-object event during setup")
    if event.get("type") == "Error":
        raise RuntimeError(f"Deepgram rejected the session: {event}")
    return event


async def initialize_deepgram(
    deepgram_ws: ClientConnection,
    settings: dict[str, Any],
) -> None:
    welcome = await _receive_deepgram_event(deepgram_ws)
    if welcome.get("type") != "Welcome":
        raise RuntimeError(
            f"Expected Deepgram Welcome before Settings; received {welcome}"
        )

    await deepgram_ws.send(json.dumps(settings))
    while True:
        event = await _receive_deepgram_event(deepgram_ws)
        event_type = event.get("type")
        if event_type == "SettingsApplied":
            logger.info("Deepgram voice agent settings applied")
            return
        if event_type == "Warning":
            logger.warning("Deepgram warning during setup: %s", event)
        elif event_type != "Welcome":
            raise RuntimeError(
                f"Expected Deepgram SettingsApplied; received {event}"
            )


async def handle_barge_in(
    event: dict[str, Any],
    exotel_ws: ServerConnection,
    stream_sid: str,
) -> None:
    if event.get("type") == "UserStartedSpeaking":
        logger.info("Caller started speaking; clearing Exotel playback")
        await exotel_ws.send(
            json.dumps({"event": "clear", "stream_sid": stream_sid})
        )


def create_function_call_response(
    function_id: str,
    function_name: str,
    result: Mapping[str, object],
    thought_signature: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "type": "FunctionCallResponse",
        "id": function_id,
        "name": function_name,
        "content": json.dumps(result),
    }
    if thought_signature is not None:
        response["thought_signature"] = thought_signature
    return response


def validate_function_arguments(
    function_name: str,
    arguments: dict[str, Any],
) -> dict[str, str]:
    if function_name not in FUNCTION_MAP:
        raise ValueError(f"Unknown function: {function_name}")

    if function_name == "submit_customer_enquiry":
        missing = [field for field in ENQUIRY_FIELDS if field not in arguments]
        invalid = [
            field for field in ENQUIRY_FIELDS
            if field in arguments and not isinstance(arguments[field], str)
        ]
        if missing or invalid:
            problems = []
            if missing:
                problems.append(f"missing fields: {', '.join(missing)}")
            if invalid:
                problems.append(f"fields must be strings: {', '.join(invalid)}")
            raise ValueError("; ".join(problems))

    return {field: arguments[field] for field in ENQUIRY_FIELDS}


async def send_pending_enquiries(
    pending_enquiries: dict[str, dict[str, str]],
) -> None:
    for function_id, enquiry in list(pending_enquiries.items()):
        pending_enquiries.pop(function_id)
        try:
            result = await asyncio.to_thread(
                FUNCTION_MAP["submit_customer_enquiry"],
                enquiry,
            )
        except (ValueError, OSError, smtplib.SMTPException) as exc:
            logger.error(
                "Failed to send enquiry email after call %s: %s",
                function_id,
                exc,
            )
        else:
            if result.get("success") is not True:
                logger.error(
                    "Enquiry email was not sent after call %s: %s",
                    function_id,
                    result.get("error", result.get("message", "unknown email failure")),
                )
            else:
                logger.info(
                    "Enquiry email sent after call %s: %s",
                    function_id,
                    result.get("message", "success"),
                )


async def handle_function_call_request(
    event: dict[str, Any],
    deepgram_ws: ClientConnection,
    pending_enquiries: dict[str, dict[str, str]],
) -> None:
    function_calls = event.get("functions")
    if not isinstance(function_calls, list):
        logger.warning("Ignoring Deepgram FunctionCallRequest without a functions list")
        return

    for function_call in function_calls:
        if not isinstance(function_call, dict):
            logger.warning("Ignoring malformed Deepgram function call entry")
            continue

        function_id = function_call.get("id")
        function_name = function_call.get("name")
        if not isinstance(function_id, str) or not isinstance(function_name, str):
            logger.warning("Ignoring function call without a valid ID and name")
            continue
        if function_call.get("client_side") is not True:
            logger.info("Skipping non-client-side function call: %s", function_name)
            continue

        thought_signature = function_call.get("thought_signature")
        if not isinstance(thought_signature, str):
            thought_signature = None

        logger.info("Deepgram requested function: %s (%s)", function_name, function_id)
        try:
            raw_arguments = function_call.get("arguments")
            if not isinstance(raw_arguments, str):
                raise ValueError("function arguments must be a JSON string")
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("function arguments must decode to a JSON object")
            enquiry = validate_function_arguments(
                function_name,
                arguments,
            )
            pending_enquiries[function_id] = enquiry
            result: Mapping[str, object] = {"success": True}
            logger.info(
                "Staged enquiry for email after call ends: %s (%s)",
                function_name,
                function_id,
            )
        except (json.JSONDecodeError, ValueError, OSError, smtplib.SMTPException) as exc:
            logger.error(
                "Deepgram function call failed: %s (%s): %s",
                function_name,
                function_id,
                exc,
            )
            result = {"success": False, "error": str(exc)}

        response = create_function_call_response(
            function_id,
            function_name,
            result,
            thought_signature,
        )
        await deepgram_ws.send(json.dumps(response))
        logger.info("Function result sent to Deepgram: %s (%s)", function_name, function_id)


async def handle_text_message(
    event: dict[str, Any],
    exotel_ws: ServerConnection,
    deepgram_ws: ClientConnection,
    stream_sid: str,
    pending_enquiries: dict[str, dict[str, str]],
) -> None:
    await handle_barge_in(event, exotel_ws, stream_sid)
    event_type = event.get("type")
    if event_type == "FunctionCallRequest":
        await handle_function_call_request(
            event,
            deepgram_ws,
            pending_enquiries,
        )
    elif event_type == "FunctionCallCancelled":
        cancelled = event.get("functions")
        if not isinstance(cancelled, list):
            logger.warning(
                "Ignoring Deepgram FunctionCallCancelled without a functions list"
            )
            return
        for function_call in cancelled:
            if not isinstance(function_call, dict):
                continue
            function_id = function_call.get("id")
            if isinstance(function_id, str) and pending_enquiries.pop(
                function_id,
                None,
            ) is not None:
                logger.info(
                    "Discarded cancelled enquiry; no email will be sent: %s",
                    function_id,
                )


async def deepgram_sender(
    deepgram_ws: ClientConnection,
    audio_queue: asyncio.Queue[QueuedAudio],
) -> None:
    loop = asyncio.get_running_loop()
    chunk_count = 0
    total_queue_wait = 0.0
    max_queue_wait = 0.0
    total_send_time = 0.0
    max_send_time = 0.0

    while True:
        queued_audio = await audio_queue.get()
        queue_wait = loop.time() - queued_audio.received_at
        send_started_at = loop.time()
        await deepgram_ws.send(queued_audio.payload)
        send_time = loop.time() - send_started_at

        chunk_count += 1
        total_queue_wait += queue_wait
        max_queue_wait = max(max_queue_wait, queue_wait)
        total_send_time += send_time
        max_send_time = max(max_send_time, send_time)
        if chunk_count == AUDIO_RELAY_LOG_INTERVAL:
            logger.info(
                "Audio relay timing (last %s chunks): "
                "average_queue_wait=%.3fs max_queue_wait=%.3fs "
                "average_deepgram_send=%.3fs max_deepgram_send=%.3fs "
                "queue_depth=%s/%s",
                chunk_count,
                total_queue_wait / chunk_count,
                max_queue_wait,
                total_send_time / chunk_count,
                max_send_time,
                audio_queue.qsize(),
                audio_queue.maxsize,
            )
            chunk_count = 0
            total_queue_wait = 0.0
            max_queue_wait = 0.0
            total_send_time = 0.0
            max_send_time = 0.0


async def deepgram_receiver(
    deepgram_ws: ClientConnection,
    exotel_ws: ServerConnection,
    stream_sid_future: asyncio.Future[str],
    pending_enquiries: dict[str, dict[str, str]],
) -> None:
    stream_sid = await stream_sid_future
    async for message in deepgram_ws:
        if isinstance(message, str):
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid JSON event from Deepgram")
                continue
            if not isinstance(event, dict):
                logger.warning("Ignoring non-object event from Deepgram")
                continue

            event_type = event.get("type")
            if event_type == "Error":
                raise RuntimeError(f"Deepgram voice agent error: {event}")
            if isinstance(event_type, str):
                logger.info("Received Deepgram event: %s", event_type)
            if event_type == "LatencyReport":
                latency_fields = {
                    key: value
                    for key, value in event.items()
                    if key != "type"
                }
                logger.info(
                    "Deepgram turn latency (seconds): %s",
                    json.dumps(latency_fields),
                )
            await handle_text_message(
                event,
                exotel_ws,
                deepgram_ws,
                stream_sid,
                pending_enquiries,
            )
            if event_type == "Warning":
                logger.warning("Deepgram voice agent warning: %s", event)
            continue

        media_message = {
            "event": "media",
            "stream_sid": stream_sid,
            "media": {"payload": base64.b64encode(message).decode("ascii")},
        }
        await exotel_ws.send(json.dumps(media_message))


async def exotel_receiver(
    exotel_ws: ServerConnection,
    audio_queue: asyncio.Queue[QueuedAudio],
    stream_sid_future: asyncio.Future[str],
    pending_enquiries: dict[str, dict[str, str]],
) -> None:
    stream_sid: str | None = None

    try:
        async for message in exotel_ws:
            try:
                event = json.loads(message)
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                logger.warning("Ignoring invalid JSON message from Exotel")
                continue

            if not isinstance(event, dict):
                logger.warning("Ignoring non-object message from Exotel")
                continue

            event_name = event.get("event")
            if event_name == "connected":
                logger.info("Exotel WebSocket connected")
            elif event_name == "start":
                start = event.get("start", {})
                if not isinstance(start, dict):
                    raise ValueError("Exotel start event contains invalid start data")
                candidate = (
                    start.get("stream_sid")
                    or start.get("streamSid")
                    or event.get("stream_sid")
                    or event.get("streamSid")
                )
                if not isinstance(candidate, str) or not candidate:
                    raise ValueError("Exotel start event did not contain a stream SID")
                stream_sid = candidate
                if stream_sid_future.done():
                    if stream_sid_future.result() != stream_sid:
                        raise ValueError("Exotel changed stream SID during a call")
                else:
                    stream_sid_future.set_result(stream_sid)
                logger.info("Exotel stream started: %s", stream_sid)
                media_format = start.get("media_format") or start.get("mediaFormat")
                if isinstance(media_format, dict):
                    logger.info("Exotel audio format: %s", json.dumps(media_format))
            elif event_name == "media":
                media = event.get("media")
                if not isinstance(media, dict):
                    logger.warning("Ignoring Exotel media event with invalid data")
                    continue
                payload = media.get("payload")
                if not isinstance(payload, str) or not payload:
                    logger.warning("Ignoring Exotel media event without an audio payload")
                    continue
                if stream_sid is None:
                    logger.warning("Ignoring Exotel media event received before start")
                    continue
                try:
                    audio = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    logger.warning("Ignoring Exotel media event with invalid base64 audio")
                    continue
                if (
                    audio_queue.maxsize > 0
                    and audio_queue.qsize() >= int(audio_queue.maxsize * 0.8)
                ):
                    logger.warning(
                        "Caller audio queue is backing up: %s/%s chunks",
                        audio_queue.qsize(),
                        audio_queue.maxsize,
                    )
                await audio_queue.put(QueuedAudio(audio, asyncio.get_running_loop().time()))
            elif event_name == "stop":
                logger.info("Exotel stream stopped: %s", stream_sid or "unknown")
                break
            elif event_name == "dtmf":
                logger.info("Exotel DTMF event received")
    except ConnectionClosed as exc:
        logger.info("Exotel WebSocket disconnected: %s", exc)
    await send_pending_enquiries(pending_enquiries)


async def exotel_handler(exotel_ws: ServerConnection) -> None:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    logger.info("Exotel call connected; preparing Deepgram voice agent")
    settings = load_config()
    audio_queue: asyncio.Queue[QueuedAudio] = asyncio.Queue(maxsize=QUEUE_SIZE)
    stream_sid_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    pending_enquiries: dict[str, dict[str, str]] = {}
    exotel_task = asyncio.create_task(
        exotel_receiver(
            exotel_ws,
            audio_queue,
            stream_sid_future,
            pending_enquiries,
        )
    )
    reconnect_attempt = 0

    try:
        while not exotel_task.done():
            try:
                async with connect(
                    DEEPGRAM_URL,
                    additional_headers={"Authorization": f"Token {api_key}"},
                    open_timeout=HANDSHAKE_TIMEOUT_SECONDS,
                    ping_interval=20,
                    ping_timeout=20,
                ) as deepgram_ws:
                    logger.info("Connected to Deepgram voice agent")
                    await initialize_deepgram(deepgram_ws, settings)
                    reconnect_attempt = 0
                    tasks = [
                        asyncio.create_task(deepgram_sender(deepgram_ws, audio_queue)),
                        asyncio.create_task(
                            deepgram_receiver(
                                deepgram_ws,
                                exotel_ws,
                                stream_sid_future,
                                pending_enquiries,
                            )
                        ),
                    ]

                    try:
                        done, _ = await asyncio.wait(
                            [*tasks, exotel_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if exotel_task in done:
                            exotel_task.result()
                            return
                        for task in done:
                            task.result()
                        raise ConnectionError(
                            "Deepgram session ended before the Exotel call"
                        )
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except (ConnectionClosed, OSError, TimeoutError) as exc:
                if exotel_task.done():
                    exotel_task.result()
                    return

                delay = min(2**reconnect_attempt, MAX_RECONNECT_DELAY_SECONDS)
                reconnect_attempt += 1
                logger.warning(
                    "Deepgram connection interrupted: %s. Retrying in %s seconds "
                    "(attempt %s)",
                    exc,
                    delay,
                    reconnect_attempt,
                )
                done, _ = await asyncio.wait({exotel_task}, timeout=delay)
                if done:
                    exotel_task.result()
                    return
    finally:
        if not exotel_task.done():
            exotel_task.cancel()
        await asyncio.gather(exotel_task, return_exceptions=True)


async def _serve() -> None:
    port = int(os.getenv("PORT", "5001"))
    async with serve(exotel_handler, "0.0.0.0", port):
        logger.info("Exotel to Deepgram WebSocket server listening on port %s", port)
        await asyncio.Future()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        logger.info("WebSocket server stopped")
