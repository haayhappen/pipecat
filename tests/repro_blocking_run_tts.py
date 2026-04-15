#!/usr/bin/env python3
"""
REPRODUCTION: Azure-style blocking run_tts deadlocks; ElevenLabs-style does not.

Run:  python tests/repro_blocking_run_tts.py

Demonstrates two mock TTS services that mirror the real implementations:

  AzureStyleTTS  — run_tts blocks on asyncio.Queue.get() waiting for SDK
                   callbacks (see azure/tts.py line 728-729).
                   Pipeline DEADLOCKS when SDK never calls back.

  ElevenLabsStyleTTS — run_tts sends text over WebSocket and immediately
                       yields None (see elevenlabs/tts.py line 885).
                       Audio arrives asynchronously in a receive task.
                       Pipeline completes normally even if no audio arrives.

Expected output:
  ElevenLabs-style completes instantly.
  Azure-style hangs forever (killed by 8 s timeout).

Root cause (compound bug):

  1. AzureTTSService.run_tts (line 728-729 of azure/tts.py) blocks inside
     tts_process_generator.  If the SDK silently fails, process_frame never
     returns, and EndFrame is stuck in the process queue.

  2. User interruptions CANNOT break this deadlock: InterruptionFrame is a
     SystemFrame (processed on a separate task), but _start_interruption
     (frame_processor.py line 826) refuses to cancel the process task when
     an UninterruptibleFrame (EndFrame) is queued.  It only resets the queue,
     leaving the stuck process task running.

  3. The pipeline is genuinely deadlocked — no external signal can rescue it.

During normal conversation (no EndFrame queued), interruptions DO cancel the
stuck process task.  The issue is specific to end-of-call flows.

ElevenLabsTTSService.run_tts (line 885 of elevenlabs/tts.py) yields None
immediately, so tts_process_generator returns in microseconds.  Audio
delivery and context cleanup happen in a separate receive task.
"""

import asyncio
import os
import sys
import time
from typing import AsyncGenerator, Optional

os.environ["LOGURU_LEVEL"] = "CRITICAL"

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService

_SAMPLE_RATE = 16000
_FAKE_AUDIO = b"\x00\x01" * 320


# ---------------------------------------------------------------------------
# Mock: Azure-style (blocks in run_tts)
# ---------------------------------------------------------------------------

class AzureStyleTTS(TTSService):
    """Mirrors AzureTTSService.run_tts: blocks on asyncio.Queue.get().

    See pipecat/services/azure/tts.py lines 728-729:
        while True:
            chunk = await self._audio_queue.get()
    """

    def __init__(self):
        super().__init__(
            push_start_frame=True,   # same as Azure
            push_stop_frames=True,   # same as Azure
            pause_frame_processing=True,  # same as Azure
            sample_rate=_SAMPLE_RATE,
        )
        self._audio_queue: asyncio.Queue = asyncio.Queue()

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Exactly what Azure does: block waiting for SDK callback audio.
        # If SDK never calls back → blocks forever.
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            yield TTSAudioRawFrame(
                audio=chunk, sample_rate=_SAMPLE_RATE, num_channels=1,
                context_id=context_id,
            )


# ---------------------------------------------------------------------------
# Mock: ElevenLabs-style (non-blocking run_tts)
# ---------------------------------------------------------------------------

class ElevenLabsStyleTTS(TTSService):
    """Mirrors ElevenLabsTTSService.run_tts: yields None immediately.

    See pipecat/services/elevenlabs/tts.py line 885:
        yield None

    Audio arrives asynchronously via a separate WebSocket receive task.
    run_tts returns in microseconds, freeing tts_process_generator.
    """

    def __init__(self):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,  # same as ElevenLabs
            sample_rate=_SAMPLE_RATE,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # ElevenLabs creates the audio context, sends text over websocket,
        # then immediately yields None.  Audio delivery happens in a
        # background _receive_task via append_to_audio_context.
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)

        # Simulate: fire-and-forget text send, audio arrives async.
        # In real ElevenLabs this is: await self._send_text(text, ctx)
        async def _deliver_audio_later():
            await asyncio.sleep(0.1)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE,
                    num_channels=1, context_id=context_id,
                ),
            )
            await self.append_to_audio_context(
                context_id, TTSStoppedFrame(context_id=context_id)
            )
            await self.remove_audio_context(context_id)

        self.create_task(_deliver_audio_later(), name="mock_el_deliver")

        yield None  # <-- THE KEY DIFFERENCE: returns immediately


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Sink(FrameProcessor):
    """Echoes BotStoppedSpeakingFrame upstream (like a real transport)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


async def run_with_tts(label: str, tts: TTSService, timeout: float) -> bool:
    """Run a pipeline with TTSSpeakFrame + EndFrame.  Returns True if completed."""
    pipeline = Pipeline([tts, Sink()])
    task = PipelineTask(pipeline, params=PipelineParams())
    completed = asyncio.Event()

    async def push():
        await asyncio.sleep(0.1)
        await task.queue_frame(TTSSpeakFrame(text="Auf Wiederhören!"))
        await task.queue_frame(EndFrame())

    async def run_all():
        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push())
        completed.set()

    bg = asyncio.create_task(run_all())
    t0 = time.monotonic()

    try:
        await asyncio.wait_for(completed.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass

    elapsed = time.monotonic() - t0
    ok = completed.is_set()

    if ok:
        print(f"  [{label}] Completed in {elapsed:.2f}s ✓", flush=True)
    else:
        print(f"  [{label}] DEADLOCKED — no completion after {elapsed:.1f}s ✗", flush=True)

    bg.cancel()
    try:
        await bg
    except (asyncio.CancelledError, Exception):
        pass

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70, flush=True)
    print("Reproduction: Azure-style vs ElevenLabs-style run_tts", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # --- ElevenLabs-style (should complete) ---
    print("1) ElevenLabs-style TTS (non-blocking run_tts, yields None):", flush=True)
    el_ok = await run_with_tts("ElevenLabs", ElevenLabsStyleTTS(), timeout=5.0)

    print(flush=True)

    # --- Azure-style (should deadlock) ---
    print("2) Azure-style TTS (blocking run_tts, awaits queue.get()):", flush=True)
    az_ok = await run_with_tts("Azure", AzureStyleTTS(), timeout=5.0)

    print(flush=True)
    print("=" * 70, flush=True)
    if el_ok and not az_ok:
        print("BUG CONFIRMED: ElevenLabs-style completes, Azure-style deadlocks.", flush=True)
    elif el_ok and az_ok:
        print("Both completed — bug may be fixed!", flush=True)
    else:
        print(f"Unexpected result: ElevenLabs={el_ok}, Azure={az_ok}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
