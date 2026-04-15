#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user idle detection."""

import asyncio
import os
import time
from typing import Optional

from loguru import logger as _idle_logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class UserIdleController(BaseObject):
    """Controller for managing user idle detection.

    This class monitors user activity and triggers an event when the user has been
    idle (not speaking) for a configured timeout period after the bot finishes
    speaking. The timer starts when BotStoppedSpeakingFrame is received and is
    cancelled when someone starts speaking again (UserStartedSpeakingFrame or
    BotStartedSpeakingFrame).

    The timer is suppressed while a user turn is in progress to avoid false
    triggers during interruptions (where BotStoppedSpeakingFrame arrives while
    the user is still speaking).

    Event handlers available:

    - on_user_turn_idle: Emitted when the user has been idle for the timeout period.

    Example::

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            # Handle user idle - send reminder, prompt, etc.
            ...
    """

    def __init__(
        self,
        *,
        user_idle_timeout: float = 0,
    ):
        """Initialize the user idle controller.

        Args:
            user_idle_timeout: Timeout in seconds before considering the user idle.
                0 disables idle detection.
        """
        super().__init__()

        self._user_idle_timeout = user_idle_timeout

        self._task_manager: Optional[BaseTaskManager] = None

        self._user_turn_in_progress: bool = False
        self._function_calls_in_progress: int = 0
        self._idle_timer_task: Optional[asyncio.Task] = None

        self._diag_enabled = os.getenv("DEAF_PIPELINE_DIAGNOSTICS", "false").lower() == "true"
        self._diag_timer_start_reason: str | None = None
        self._diag_timer_cancel_reason: str | None = None
        self._diag_timer_started_at: float = 0.0

        self._register_event_handler("on_user_turn_idle", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user idle controller was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the controller with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()
        await self._cancel_idle_timer()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to track user activity state.

        Args:
            frame: The frame to be processed.
        """
        if isinstance(frame, UserIdleTimeoutUpdateFrame):
            self._user_idle_timeout = frame.timeout
            if self._user_idle_timeout <= 0:
                await self._cancel_idle_timer()
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            if not self._user_turn_in_progress and self._function_calls_in_progress == 0:
                self._diag_timer_start_reason = "bot_stopped_speaking"
                await self._start_idle_timer()
            elif self._diag_enabled:
                _idle_logger.debug(
                    f"[IDLE-DIAG] BotStoppedSpeaking but timer NOT started "
                    f"(user_turn={self._user_turn_in_progress}, fc={self._function_calls_in_progress})"
                )
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._diag_timer_cancel_reason = "bot_started_speaking"
            await self._cancel_idle_timer()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_turn_in_progress = True
            self._diag_timer_cancel_reason = "user_started_speaking"
            await self._cancel_idle_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_in_progress = False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._function_calls_in_progress += len(frame.function_calls)
            self._diag_timer_cancel_reason = "function_calls_started"
            await self._cancel_idle_timer()
        elif isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self._function_calls_in_progress = max(0, self._function_calls_in_progress - 1)

    def get_state(self) -> dict:
        """Return a snapshot of the idle controller's internal state for diagnostics."""
        timer_running = self._idle_timer_task is not None
        timer_age_s = None
        if timer_running and self._diag_timer_started_at:
            timer_age_s = round(time.monotonic() - self._diag_timer_started_at, 3)
        return {
            "timer_running": timer_running,
            "timer_age_s": timer_age_s,
            "timer_timeout_s": self._user_idle_timeout,
            "timer_start_reason": self._diag_timer_start_reason,
            "timer_last_cancel_reason": self._diag_timer_cancel_reason,
            "user_turn_in_progress": self._user_turn_in_progress,
            "function_calls_in_progress": self._function_calls_in_progress,
        }

    async def _start_idle_timer(self):
        """Start (or restart) the idle timer."""
        if self._user_idle_timeout <= 0:
            return
        await self._cancel_idle_timer()
        self._diag_timer_started_at = time.monotonic()
        if self._diag_enabled:
            _idle_logger.debug(
                f"[IDLE-DIAG] Timer started ({self._user_idle_timeout}s) "
                f"reason={self._diag_timer_start_reason}"
            )
        self._idle_timer_task = self.task_manager.create_task(
            self._idle_timer_expired(),
            f"{self}::idle_timer",
        )
        await asyncio.sleep(0)

    async def _cancel_idle_timer(self):
        """Cancel the idle timer if running."""
        if self._idle_timer_task:
            elapsed = round(time.monotonic() - self._diag_timer_started_at, 3) if self._diag_timer_started_at else 0
            if self._diag_enabled:
                _idle_logger.debug(
                    f"[IDLE-DIAG] Timer cancelled after {elapsed}s "
                    f"reason={self._diag_timer_cancel_reason}"
                )
            await self.task_manager.cancel_task(self._idle_timer_task)
            self._idle_timer_task = None

    async def _idle_timer_expired(self):
        """Sleep for the timeout duration then fire the idle event."""
        await asyncio.sleep(self._user_idle_timeout)
        elapsed = round(time.monotonic() - self._diag_timer_started_at, 3) if self._diag_timer_started_at else 0
        if self._diag_enabled:
            _idle_logger.debug(
                f"[IDLE-DIAG] Timer expired after {elapsed}s "
                f"(timeout={self._user_idle_timeout}s, start_reason={self._diag_timer_start_reason})"
            )
        self._idle_timer_task = None
        await self._call_event_handler("on_user_turn_idle")
