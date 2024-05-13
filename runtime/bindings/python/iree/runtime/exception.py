# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class ErrorUnavailable(RuntimeError):
    """
    The system used to perform the operation is currently (and transiently)
    unavailable. Callers can retry with backoff.
    """

    def __init__(self, message):
        super().__init__(self, message)
