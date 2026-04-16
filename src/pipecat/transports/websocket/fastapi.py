#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FastAPI WebSocket transport implementation for Pipecat.

This module provides WebSocket-based transport for real-time audio/video streaming
using FastAPI and WebSocket connections. Supports binary and text serialization
with configurable session timeouts and WAV header generation.
"""

import asyncio
import io
import os
import time
import traceback
import typing
import wave
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use FastAPI websockets, you need to `pip install pipecat-ai[websocket]`."
    )
    raise Exception(f"Missing module: {e}")


# Threshold above which an individual WebSocket send() call is considered "slow"
# and a WARN is emitted. The event loop handing a call that long strongly
# suggests TCP back-pressure or a half-dead remote. Read-only diagnostic, does
# not change behaviour.
SLOW_SEND_THRESHOLD_MS: float = 250.0

# Silence watchdog configuration. The watchdog is gated behind the
# DEAF_PIPELINE_DIAGNOSTICS environment variable and NEVER cancels the receive
# task or closes the socket. It only logs.
_DEAF_PIPELINE_DIAGNOSTICS_ENV = "DEAF_PIPELINE_DIAGNOSTICS"
SILENCE_WATCHDOG_INTERVAL_S: float = 10.0
SILENCE_WATCHDOG_THRESHOLD_S: float = 10.0


def _deaf_diagnostics_enabled() -> bool:
    return os.getenv(_DEAF_PIPELINE_DIAGNOSTICS_ENV, "false").lower() == "true"


class FastAPIWebsocketParams(TransportParams):
    """Configuration parameters for FastAPI WebSocket transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for encoding/decoding messages.
        session_timeout: Session timeout in seconds, None for no timeout.
        fixed_audio_packet_size: Optional fixed-size packetization for raw PCM audio payloads.
            Useful when the remote WebSocket media endpoint requires strict audio framing.
    """

    add_wav_header: bool = False
    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None
    fixed_audio_packet_size: Optional[int] = None


class FastAPIWebsocketCallbacks(BaseModel):
    """Callback functions for WebSocket events.

    Parameters:
        on_client_connected: Called when a client connects to the WebSocket.
        on_client_disconnected: Called when a client disconnects from the WebSocket.
        on_session_timeout: Called when a session timeout occurs.
    """

    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
    on_session_timeout: Callable[[WebSocket], Awaitable[None]]


class _WebSocketMessageIterator:
    """Async iterator for WebSocket messages that yields both binary and text."""

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes | str:
        message = await self._websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise StopAsyncIteration
        if "bytes" in message and message["bytes"] is not None:
            return message["bytes"]
        if "text" in message and message["text"] is not None:
            return message["text"]
        raise StopAsyncIteration


class FastAPIWebsocketClient:
    """WebSocket client wrapper for handling connections and message passing.

    Manages WebSocket state, message sending/receiving, and connection lifecycle
    with support for both binary and text message types.
    """

    def __init__(self, websocket: WebSocket, callbacks: FastAPIWebsocketCallbacks):
        """Initialize the WebSocket client.

        Args:
            websocket: The FastAPI WebSocket connection.
            callbacks: Event callback functions.
        """
        self._websocket = websocket
        self._closing = False
        self._callbacks = callbacks
        self._leave_counter = 0

        # Egress diagnostic counters (read-only telemetry).
        self._diag_send_attempts: int = 0
        self._diag_send_success: int = 0
        self._diag_send_bytes: int = 0
        self._diag_last_send_ts: float = 0.0
        self._diag_last_send_success_ts: float = 0.0
        self._diag_send_exc_count: int = 0
        self._diag_last_send_exc_ts: float = 0.0
        self._diag_last_send_exc_type: Optional[str] = None
        # Send-path wall-clock latency (see SLOW_SEND_THRESHOLD_MS).
        self._diag_last_send_latency_ms: float = 0.0
        self._diag_max_send_latency_ms: float = 0.0
        self._diag_slow_send_count: int = 0
        # Non-zero while a send() is awaiting; used by the silence watchdog to
        # detect sends blocked on TCP drain.
        self._diag_in_flight_send_started_ts: float = 0.0
        # Number of send() calls that were skipped because _can_send() returned
        # False. A growing value while the pipeline still pushes frames
        # indicates the WS layer believes we're disconnected.
        self._diag_send_skipped_count: int = 0

    async def setup(self, _: StartFrame):
        """Set up the WebSocket client.

        Args:
            _: The start frame (unused).
        """
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        """Get an async iterator for receiving WebSocket messages.

        Returns:
            An async iterator yielding bytes or strings.
        """
        return _WebSocketMessageIterator(self._websocket)

    async def send(self, data: str | bytes):
        """Send data through the WebSocket connection.

        Args:
            data: The data to send (string or bytes).
        """
        if not self._can_send():
            self._diag_send_skipped_count += 1
            return

        self._diag_send_attempts += 1
        self._diag_last_send_ts = time.monotonic()
        data_len = len(data) if isinstance(data, (bytes, bytearray, str)) else 0
        t0 = time.monotonic()
        self._diag_in_flight_send_started_ts = t0
        try:
            if isinstance(data, bytes):
                await self._websocket.send_bytes(data)
            else:
                await self._websocket.send_text(data)
            self._diag_send_success += 1
            self._diag_send_bytes += data_len
            self._diag_last_send_success_ts = time.monotonic()
        except Exception as e:
            self._diag_send_exc_count += 1
            self._diag_last_send_exc_ts = time.monotonic()
            self._diag_last_send_exc_type = e.__class__.__name__
            logger.warning(
                f"{self} exception sending data: {e.__class__.__name__} ({e}), "
                f"application_state: {self._websocket.application_state}, "
                f"send_attempts={self._diag_send_attempts}, "
                f"send_exc_count={self._diag_send_exc_count}"
            )
        finally:
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._diag_last_send_latency_ms = latency_ms
            if latency_ms > self._diag_max_send_latency_ms:
                self._diag_max_send_latency_ms = latency_ms
            if latency_ms > SLOW_SEND_THRESHOLD_MS:
                self._diag_slow_send_count += 1
                try:
                    client_state = self._websocket.client_state.name
                    application_state = self._websocket.application_state.name
                except Exception:
                    client_state = "unknown"
                    application_state = "unknown"
                logger.warning(
                    f"{self} slow websocket send: {latency_ms:.1f} ms "
                    f"(threshold={SLOW_SEND_THRESHOLD_MS:.0f} ms, "
                    f"bytes={data_len}, client_state={client_state}, "
                    f"application_state={application_state}, "
                    f"slow_send_count={self._diag_slow_send_count})"
                )
            self._diag_in_flight_send_started_ts = 0.0

    def get_egress_stats(self) -> dict:
        """Return outbound-send diagnostic counters.

        The stats distinguish three fatal failure modes of a silently-dead
        WebSocket: fast-success (TCP buffer absorbing writes), blocked-drain
        (in_flight_send_age_s climbing), and raised-exception.
        """
        now = time.monotonic()
        in_flight_age = (
            round(now - self._diag_in_flight_send_started_ts, 3)
            if self._diag_in_flight_send_started_ts
            else None
        )
        try:
            client_state = self._websocket.client_state.name
            application_state = self._websocket.application_state.name
        except Exception:
            client_state = "unknown"
            application_state = "unknown"
        return {
            "send_attempts": self._diag_send_attempts,
            "send_success": self._diag_send_success,
            "send_bytes": self._diag_send_bytes,
            "send_exc_count": self._diag_send_exc_count,
            "send_skipped_count": self._diag_send_skipped_count,
            "last_send_age_s": round(now - self._diag_last_send_ts, 3) if self._diag_last_send_ts else None,
            "last_send_success_age_s": (
                round(now - self._diag_last_send_success_ts, 3)
                if self._diag_last_send_success_ts
                else None
            ),
            "last_send_exc_age_s": (
                round(now - self._diag_last_send_exc_ts, 3)
                if self._diag_last_send_exc_ts
                else None
            ),
            "last_send_exc_type": self._diag_last_send_exc_type,
            "last_send_latency_ms": round(self._diag_last_send_latency_ms, 2),
            "max_send_latency_ms": round(self._diag_max_send_latency_ms, 2),
            "slow_send_count": self._diag_slow_send_count,
            "in_flight_send_age_s": in_flight_age,
            "is_connected": self.is_connected,
            "is_closing": self.is_closing,
            "client_state": client_state,
            "application_state": application_state,
        }

    async def disconnect(self):
        """Disconnect the WebSocket client."""
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            try:
                await self._websocket.close()
            except Exception as e:
                logger.error(f"{self} exception while closing the websocket: {e}")

    async def trigger_client_disconnected(self):
        """Trigger the client disconnected callback."""
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        """Trigger the client connected callback."""
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timeout(self):
        """Trigger the client timeout callback."""
        await self._callbacks.on_session_timeout(self._websocket)

    def _can_send(self):
        """Check if data can be sent through the WebSocket."""
        return (
            self.is_connected
            and not self.is_closing
            and self._websocket.application_state != WebSocketState.DISCONNECTED
        )

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is currently connected.

        Returns:
            True if the WebSocket is in connected state.
        """
        return self._websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_closing(self) -> bool:
        """Check if the WebSocket is currently closing.

        Returns:
            True if the WebSocket is in the process of closing.
        """
        return self._closing


class FastAPIWebsocketInputTransport(BaseInputTransport):
    """Input transport for FastAPI WebSocket connections.

    Handles incoming WebSocket messages, deserializes frames, and manages
    connection monitoring with optional session timeouts.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket input transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._monitor_websocket_task = None

        self._initialized = False

        self._diag_media_in_count: int = 0
        self._diag_last_media_in_ts: float = 0.0
        self._diag_non_audio_msg_count: int = 0
        # Unconditional: stamped on every WS iteration (any kind of message).
        # This is the primary signal the silence watchdog uses.
        self._diag_last_any_recv_ts: float = 0.0
        # Stamped whenever we receive a non-audio WS message (text, empty, or
        # non-InputAudioRawFrame after deserialization).
        self._diag_last_non_audio_ts: float = 0.0
        self._diag_last_non_audio_kind: Optional[str] = None
        # Soft silence watchdog (gated on DEAF_PIPELINE_DIAGNOSTICS).
        self._silence_watchdog_task = None
        # Receive loop exit logging: set to non-None when finally-block runs,
        # used by pipeline-level diagnostics to correlate with heartbeats.
        self._diag_receive_loop_exit_ts: float = 0.0
        self._diag_receive_loop_exit_reason: Optional[str] = None

    async def start(self, frame: StartFrame):
        """Start the input transport and begin message processing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(frame)
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._monitor_websocket_task and self._params.session_timeout:
            self._monitor_websocket_task = self.create_task(self._monitor_websocket())
        await self._client.trigger_client_connected()
        await self.push_frame(ClientConnectedFrame())
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_messages())
        if not self._silence_watchdog_task and _deaf_diagnostics_enabled():
            self._silence_watchdog_task = self.create_task(self._silence_watchdog())
        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        """Stop all running tasks."""
        if self._monitor_websocket_task:
            await self.cancel_task(self._monitor_websocket_task)
            self._monitor_websocket_task = None
        if self._silence_watchdog_task:
            await self.cancel_task(self._silence_watchdog_task)
            self._silence_watchdog_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cleanup(self):
        """Clean up transport resources."""
        await super().cleanup()
        await self._transport.cleanup()

    def get_ingress_stats(self) -> dict:
        """Return media ingress counters for diagnostics."""
        now = time.monotonic()
        return {
            "media_in_count": self._diag_media_in_count,
            "last_media_in_age_s": round(now - self._diag_last_media_in_ts, 3) if self._diag_last_media_in_ts else None,
            "non_audio_msg_count": self._diag_non_audio_msg_count,
            "last_any_recv_age_s": round(now - self._diag_last_any_recv_ts, 3) if self._diag_last_any_recv_ts else None,
            "last_non_audio_age_s": round(now - self._diag_last_non_audio_ts, 3) if self._diag_last_non_audio_ts else None,
            "last_non_audio_kind": self._diag_last_non_audio_kind,
            "receive_loop_exit_age_s": (
                round(now - self._diag_receive_loop_exit_ts, 3)
                if self._diag_receive_loop_exit_ts
                else None
            ),
            "receive_loop_exit_reason": self._diag_receive_loop_exit_reason,
        }

    async def _receive_messages(self):
        """Main message receiving loop for WebSocket messages."""
        exit_reason: str = "clean"
        try:
            async for message in self._client.receive():
                # Stamp unconditionally BEFORE serializer.deserialize so we see
                # any WS traffic (including text/control messages) reach the
                # loop, independent of whether the serializer produces a frame.
                self._diag_last_any_recv_ts = time.monotonic()

                if not self._params.serializer:
                    self._diag_non_audio_msg_count += 1
                    self._diag_last_non_audio_ts = self._diag_last_any_recv_ts
                    self._diag_last_non_audio_kind = (
                        "bytes" if isinstance(message, (bytes, bytearray)) else "text"
                    )
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    self._diag_non_audio_msg_count += 1
                    self._diag_last_non_audio_ts = time.monotonic()
                    self._diag_last_non_audio_kind = (
                        "empty-bytes"
                        if isinstance(message, (bytes, bytearray))
                        else "empty-text"
                    )
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    self._diag_media_in_count += 1
                    self._diag_last_media_in_ts = time.monotonic()
                    await self.push_audio_frame(frame)
                elif isinstance(frame, InputTransportMessageFrame):
                    self._diag_last_non_audio_ts = time.monotonic()
                    self._diag_last_non_audio_kind = "transport-message"
                    await self.broadcast_frame(InputTransportMessageFrame, message=frame.message)
                else:
                    self._diag_last_non_audio_ts = time.monotonic()
                    self._diag_last_non_audio_kind = frame.__class__.__name__
                    await self.push_frame(frame)
        except Exception as e:
            exit_reason = f"exception:{e.__class__.__name__}:{e}"
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")
        finally:
            # Explicit exit log: distinguishes a clean async-for completion
            # (Twilio sent websocket.disconnect) from a truly stuck loop that
            # this block never reaches.
            self._diag_receive_loop_exit_ts = time.monotonic()
            self._diag_receive_loop_exit_reason = exit_reason
            try:
                client_state = self._client._websocket.client_state.name
                application_state = self._client._websocket.application_state.name
            except Exception:
                client_state = "unknown"
                application_state = "unknown"
            last_media_age = (
                round(time.monotonic() - self._diag_last_media_in_ts, 2)
                if self._diag_last_media_in_ts
                else None
            )
            last_any_age = (
                round(time.monotonic() - self._diag_last_any_recv_ts, 2)
                if self._diag_last_any_recv_ts
                else None
            )
            logger.warning(
                f"{self} _receive_messages exited: "
                f"reason={exit_reason}, "
                f"client_state={client_state}, "
                f"application_state={application_state}, "
                f"is_closing={self._client.is_closing}, "
                f"media_in_count={self._diag_media_in_count}, "
                f"last_media_in_age_s={last_media_age}, "
                f"last_any_recv_age_s={last_any_age}, "
                f"non_audio_msg_count={self._diag_non_audio_msg_count}"
            )

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._client.is_closing:
            await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
        await self._client.trigger_client_timeout()

    async def _silence_watchdog(self):
        """Soft log-only watchdog for detecting silent WebSocket death.

        This task is strictly diagnostic: it NEVER cancels the receive task,
        closes the socket, or pushes frames. Its sole job is to dump the
        receive task's stack (plus ingress/egress state) when no WS message
        of any kind has arrived for ``SILENCE_WATCHDOG_THRESHOLD_S`` seconds
        AND at least one media message has been seen (so we don't fire before
        the pipeline has even stabilised).

        Gated on the ``DEAF_PIPELINE_DIAGNOSTICS`` environment variable.
        """
        diag_logger = logger.bind(diag_event="ws_silence_watchdog")
        while True:
            try:
                await asyncio.sleep(SILENCE_WATCHDOG_INTERVAL_S)
            except asyncio.CancelledError:
                raise

            if self._diag_media_in_count == 0:
                # Pipeline hasn't started receiving media yet.
                continue

            now = time.monotonic()
            last_recv = max(self._diag_last_any_recv_ts, self._diag_last_media_in_ts)
            if last_recv == 0.0:
                continue

            silence_s = now - last_recv
            if silence_s < SILENCE_WATCHDOG_THRESHOLD_S:
                continue

            ingress_stats = self.get_ingress_stats()
            try:
                egress_stats = self._client.get_egress_stats()
            except Exception as e:
                egress_stats = {"error": f"{e.__class__.__name__}: {e}"}
            try:
                audio_silence = self.get_audio_in_silence_state()
            except Exception as e:
                audio_silence = {"error": f"{e.__class__.__name__}: {e}"}

            recv_task = self._receive_task
            task_meta = {}
            stack_rendered = ""
            if recv_task is not None:
                try:
                    task_meta = {
                        "name": recv_task.get_name(),
                        "done": recv_task.done(),
                        "cancelled": recv_task.cancelled() if recv_task.done() else False,
                    }
                    frames = recv_task.get_stack(limit=20)
                    stack_rendered = "".join(traceback.format_stack(f) for f in frames)
                except Exception as e:
                    task_meta = {"stack_error": f"{e.__class__.__name__}: {e}"}

            in_flight_age = egress_stats.get("in_flight_send_age_s") if isinstance(egress_stats, dict) else None

            diag_logger.warning(
                f"{self} ws_silence_watchdog: silence_s={silence_s:.2f}, "
                f"receive_task={task_meta}, "
                f"in_flight_send_age_s={in_flight_age}, "
                f"ingress={ingress_stats}, "
                f"egress={egress_stats}, "
                f"audio_in_silence={audio_silence}\n"
                f"-- receive task stack --\n{stack_rendered}"
            )


class FastAPIWebsocketOutputTransport(BaseOutputTransport):
    """Output transport for FastAPI WebSocket connections.

    Handles outgoing frame serialization, audio streaming with timing simulation,
    and WebSocket message transmission with optional WAV header generation.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket output transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

        # Buffer for optional protocol-level audio packetization.
        # Some serializers may emit arbitrarily sized raw PCM payloads, while
        # certain downstream transports or media endpoints require audio to be
        # sent in fixed-size frames. When `params.fixed_audio_packet_size` is set,
        # this buffer accumulates outgoing audio until a full packet can be
        # emitted, preserving any remainder for subsequent sends.
        self._audio_send_buffer = bytearray()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize timing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(frame)
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cleanup(self):
        """Clean up transport resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process outgoing frames with special handling for interruptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Drop any partially buffered audio to avoid replaying stale PCM
            if self._params.fixed_audio_packet_size:
                self._audio_send_buffer.clear()

            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message frame.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebSocket with timing simulation.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if self._client.is_closing or not self._client.is_connected:
            return False

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

        return True

    async def _write_frame(self, frame: Frame):
        """Serialize and send a frame through the WebSocket."""
        if self._client.is_closing or not self._client.is_connected:
            return

        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                # Optional protocol-level audio packetization:
                # If a downstream WebSocket media endpoint requires fixed-size PCM frames,
                # configure params.fixed_audio_packet_size (e.g. 640 for 20ms @ 16kHz PCM16 mono).
                packet_bytes = self._params.fixed_audio_packet_size

                if packet_bytes and isinstance(payload, (bytes, bytearray)):
                    self._audio_send_buffer.extend(bytes(payload))

                    # Send only full frames; keep remainder for the next call.
                    while len(self._audio_send_buffer) >= packet_bytes:
                        chunk = bytes(self._audio_send_buffer[:packet_bytes])
                        del self._audio_send_buffer[:packet_bytes]
                        await self._client.send(chunk)
                    return

                await self._client.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        """Simulate audio playback timing with appropriate delays."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class FastAPIWebsocketTransport(BaseTransport):
    """FastAPI WebSocket transport for real-time audio/video streaming.

    Provides bidirectional WebSocket communication with frame serialization,
    session management, and event handling for client connections and timeouts.

    Event handlers available:

    - on_client_connected(transport, websocket): Client WebSocket connected
    - on_client_disconnected(transport, websocket): Client WebSocket disconnected
    - on_session_timeout(transport, websocket): Session timed out

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            ...
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the FastAPI WebSocket transport.

        Args:
            websocket: The FastAPI WebSocket connection.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = FastAPIWebsocketClient(websocket, self._callbacks)

        self._input = FastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> FastAPIWebsocketInputTransport:
        """Get the input transport processor.

        Returns:
            The WebSocket input transport instance.
        """
        return self._input

    def output(self) -> FastAPIWebsocketOutputTransport:
        """Get the output transport processor.

        Returns:
            The WebSocket output transport instance.
        """
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connected event."""
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnected event."""
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        """Handle session timeout event."""
        await self._call_event_handler("on_session_timeout", websocket)
