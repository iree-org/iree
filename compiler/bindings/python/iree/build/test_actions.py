# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable
from iree.build.executor import ActionConcurrency, BuildAction


class _ThunkTrampoline:
    def __init__(self, thunk, args):
        self.thunk = thunk
        self.args = args

    def __call__(self):
        self.thunk(*self.args)


class ExecuteOutOfProcessThunkAction(BuildAction):
    """Executes a callback thunk with arguments.

    Both the thunk and args must be pickleable.
    """

    def __init__(self, thunk, args, concurrency=ActionConcurrency.PROCESS, **kwargs):
        super().__init__(concurrency=concurrency, **kwargs)
        self.trampoline = _ThunkTrampoline(thunk, args)

    def _remotable_thunk(self) -> Callable[[], None]:
        return self.trampoline
