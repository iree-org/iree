# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Log sink setup for application entry points, not for library imports."""

import os
import sys

from loguru import logger

__all__ = ["configure_application_logging"]

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    " | <level>{level: <8}</level>"
    " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    " - <level>{message}</level>"
)


def configure_application_logging(
    log_output: str | os.PathLike[str] | None = None,
    level: str | int = "INFO",
    format: str = _DEFAULT_FORMAT,
) -> None:
    """Replace the application's Loguru sinks with stderr or a rotating file.

    Call once from the host application's entry point. This deliberately removes
    all existing sinks, including Loguru's default stderr sink. Applications
    that already configure Loguru should keep their own setup and only enable
    the library namespaces they need.

    Library namespaces are enabled separately, after importing the libraries.
    Importing this module does not configure logging.

    Args:
        log_output: Optional log file path. Files rotate at 10 MB. Braces in the
            path are kept as they are. If omitted, log messages are written to
            stderr, if the process has one.
        level: Minimum severity to write, defaulting to INFO.
        format: Loguru message format, defaulting to timestamps, severity,
            source location and message.
    """
    logger.remove()
    if log_output is not None:
        # Loguru fills in braces in a file name, as in "{time}.log".
        path = os.fspath(log_output).replace("{", "{{").replace("}", "}}")
        # No color codes in files, whatever the environment asks for.
        logger.add(path, format=format, level=level, rotation="10 MB", colorize=False)
    elif sys.stderr is not None:
        # Loguru decides about colors itself: it knows about terminals and the
        # environment variables that turn colors on and off.
        logger.add(sys.stderr, format=format, level=level)
