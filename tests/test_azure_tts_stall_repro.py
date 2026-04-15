#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reproduction: Azure's blocking run_tts deadlocks pipeline on end-of-call.

The bug is in AzureTTSService.run_tts (azure/tts.py lines 728-729):

    while True:
        chunk = await self._audio_queue.get()   # blocks tts_process_generator

This is a synchronous blocking loop inside an async generator.  The base
TTSService architecture runs _handle_audio_context in a separate task with a
timeout (stop_frame_timeout_s, default 3.0 s) — that timeout works correctly.
But tts_process_generator (line 1035 of tts_service.py) awaits run_tts via
``async for frame in generator``, so process_frame is blocked for as long as
run_tts blocks.  There is no mechanism to cancel run_tts when the audio
context times out.

ElevenLabsTTSService is immune because its run_tts yields None immediately
(elevenlabs/tts.py line 885) and delivers audio asynchronously via
append_to_audio_context from a background WebSocket receive task.  This means
tts_process_generator returns in microseconds.

The consequence at end-of-call (COMPOUND BUG — interruptions cannot help):

  1. end_call_gracefully queues TTSSpeakFrame("goodbye") then EndFrame.
  2. The process task picks up TTSSpeakFrame → process_frame → _push_tts_frames
     → tts_process_generator → run_tts → blocks on _audio_queue.get().
  3. EndFrame (an UninterruptibleFrame) is waiting in __process_queue.
  4. _handle_audio_context correctly times out, pushes TTSStoppedFrame.
  5. But run_tts still blocks process_frame — EndFrame cannot be dequeued.

  6. **User tries to interrupt — it DOES NOT WORK:**
     InterruptionFrame is a SystemFrame, so it IS delivered to the TTS service
     on the separate input task (not blocked by the process task).  However,
     _start_interruption (frame_processor.py line 826) checks:

         if current_is_uninterruptible or self.__process_queue.has_uninterruptible:
             self.__reset_process_queue()   # ← only cleans queue, keeps EndFrame
         else:
             await self.__cancel_process_task()  # ← would cancel stuck run_tts

     Because EndFrame (UninterruptibleFrame) is in __process_queue,
     has_uninterruptible is True, so the process task is NOT cancelled.
     The stuck run_tts continues blocking indefinitely.  Pipeline deadlocked.

During NORMAL conversation (no EndFrame queued), an InterruptionFrame WILL
cancel the process task because has_uninterruptible is False.  That's why the
issue manifests specifically at end-of-call when EndFrame is queued.

See also: repro_blocking_run_tts.py for a standalone deadlock demonstration.
"""

import asyncio
from typing import AsyncGenerator

import pytest

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

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Mock: ElevenLabs-style (non-blocking run_tts — NOT affected)
# ---------------------------------------------------------------------------


class MockElevenLabsStyleTTS(TTSService):
    """Mirrors ElevenLabsTTSService.run_tts: yields None immediately.

    Audio arrives asynchronously via append_to_audio_context from a background
    task.  tts_process_generator returns in microseconds, so process_frame is
    never blocked.
    """

    def __init__(self, audio_delay_s: float = 0.1, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            stop_frame_timeout_s=0.5,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._audio_delay_s = audio_delay_s

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)

        async def _deliver():
            await asyncio.sleep(self._audio_delay_s)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE,
                    num_channels=1, context_id=context_id,
                ),
            )
            await self.append_to_audio_context(
                context_id, TTSStoppedFrame(context_id=context_id),
            )
            await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"mock_el_{context_id}")
        yield None  # returns immediately


# ---------------------------------------------------------------------------
# Mock: Azure-style (blocking run_tts — AFFECTED)
# ---------------------------------------------------------------------------


class MockAzureStyleSlowTTS(TTSService):
    """Mirrors AzureTTSService.run_tts: blocks on asyncio.Queue.get().

    run_tts sleeps for delay_s then yields audio — simulating Azure's SDK
    taking a long time before the first synthesizing callback fires.  Because
    run_tts blocks tts_process_generator, process_frame is stuck even though
    _handle_audio_context correctly times out in its separate task.
    """

    def __init__(self, delay_s: float = 1.5, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            stop_frame_timeout_s=0.5,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )
        self._delay_s = delay_s

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Simulates: self._speech_synthesizer.speak_ssml_async(ssml)
        # then:      chunk = await self._audio_queue.get()  ← blocks here
        await asyncio.sleep(self._delay_s)
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO, sample_rate=_SAMPLE_RATE,
            num_channels=1, context_id=context_id,
        )


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


class FrameCollector(FrameProcessor):
    """Captures downstream frames; echoes BotStoppedSpeakingFrame upstream."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
            if isinstance(frame, TTSStoppedFrame):
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_azure_style_blocking_run_tts_holds_process_frame():
    """Azure-style blocking run_tts keeps process_frame stuck.

    run_tts blocks for 1.5 s.  _handle_audio_context correctly times out at
    0.5 s and pushes TTSStoppedFrame — the timeout is NOT the bug.

    The bug is that run_tts is still blocking tts_process_generator (and
    therefore process_frame) for the remaining 1.0 s.  During that time,
    EndFrame sits in the input queue unable to be processed.

    We verify: the FIRST TTSStoppedFrame (from the timeout) arrives with
    zero audio frames between TTSStartedFrame and TTSStoppedFrame.
    """
    tts = MockAzureStyleSlowTTS(delay_s=1.5)
    collector = FrameCollector()
    pipeline = Pipeline([tts, collector])
    task = PipelineTask(pipeline, params=PipelineParams())

    async def push_frames():
        await asyncio.sleep(0.05)
        await task.queue_frame(TTSSpeakFrame(text="Vielen Dank für Ihren Anruf."))
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(task), push_frames()),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        pass

    frame_types = [type(f) for f in collector.frames]
    type_names = [t.__name__ for t in frame_types]
    print(f"Downstream frames: {type_names}")

    assert TTSStartedFrame in frame_types, f"Expected TTSStartedFrame. Got: {type_names}"
    assert TTSStoppedFrame in frame_types, f"Expected TTSStoppedFrame. Got: {type_names}"

    started_idx = frame_types.index(TTSStartedFrame)
    stopped_idx = frame_types.index(TTSStoppedFrame)
    between = frame_types[started_idx + 1 : stopped_idx]

    # The timeout pushed TTSStoppedFrame before any audio — timeout works,
    # but run_tts is still blocking process_frame with no cancellation.
    assert TTSAudioRawFrame not in between, (
        f"TTSStoppedFrame arrived before any audio (timeout working correctly). "
        f"The bug is that run_tts is still blocking process_frame and nothing "
        f"cancels it.\nFrames between Started→Stopped: "
        f"{[t.__name__ for t in between]}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_elevenlabs_style_not_affected():
    """ElevenLabs-style non-blocking run_tts delivers audio correctly.

    Same stop_frame_timeout_s=0.5 as the Azure test, but run_tts yields None
    immediately.  Audio arrives asynchronously well before the timeout.
    tts_process_generator returns in microseconds, so process_frame is free
    to handle EndFrame promptly.

    This proves the issue is specific to Azure's blocking run_tts pattern.
    """
    tts = MockElevenLabsStyleTTS(audio_delay_s=0.1)
    collector = FrameCollector()
    pipeline = Pipeline([tts, collector])
    task = PipelineTask(pipeline, params=PipelineParams())

    async def push_frames():
        await asyncio.sleep(0.05)
        await task.queue_frame(TTSSpeakFrame(text="Vielen Dank für Ihren Anruf."))
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.wait_for(
        asyncio.gather(runner.run(task), push_frames()),
        timeout=10.0,
    )

    frame_types = [type(f) for f in collector.frames]
    type_names = [t.__name__ for t in frame_types]
    print(f"Downstream frames: {type_names}")

    assert TTSStartedFrame in frame_types, f"Missing TTSStartedFrame. Got: {type_names}"
    assert TTSAudioRawFrame in frame_types, f"Missing TTSAudioRawFrame. Got: {type_names}"
    assert TTSStoppedFrame in frame_types, f"Missing TTSStoppedFrame. Got: {type_names}"

    started_idx = frame_types.index(TTSStartedFrame)
    audio_idx = frame_types.index(TTSAudioRawFrame)
    stopped_idx = frame_types.index(TTSStoppedFrame)

    assert started_idx < audio_idx < stopped_idx, (
        f"Expected Started < Audio < Stopped. Got indices: "
        f"Started={started_idx}, Audio={audio_idx}, Stopped={stopped_idx}. "
        f"Frames: {type_names}"
    )
