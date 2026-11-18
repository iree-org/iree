# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.core.log_config."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from loguru import logger

from iree.core.log_config import configure_application_logging


class LogConfigTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp_path = Path(temp_dir.name)
        # Cleanups run last in, first out: the sinks, and the files they write,
        # are closed before the temporary directory is deleted.
        self.addCleanup(logger.remove)

    def test_file_sink_receives_messages_without_color_codes(self):
        log_file = self.tmp_path / "app.log"
        configure_application_logging(log_file)

        logger.info("written to the file")

        content = log_file.read_text(encoding="utf-8")
        self.assertIn("written to the file", content)
        self.assertIn("INFO", content)
        self.assertNotIn("\x1b[", content)

    def test_level_filters_messages(self):
        log_file = self.tmp_path / "app.log"
        configure_application_logging(log_file, level="WARNING")

        logger.info("too detailed")
        logger.warning("important")

        content = log_file.read_text(encoding="utf-8")
        self.assertNotIn("too detailed", content)
        self.assertIn("important", content)

    def test_format_is_applied(self):
        log_file = self.tmp_path / "app.log"
        configure_application_logging(log_file, format="{level}:{message}")

        logger.info("short")

        self.assertEqual(log_file.read_text(encoding="utf-8"), "INFO:short\n")

    def test_sinks_of_the_application_are_removed(self):
        messages: list[str] = []
        logger.add(messages.append, format="{message}", level="DEBUG")

        log_file = self.tmp_path / "app.log"
        configure_application_logging(log_file, format="{message}")
        logger.info("only to the file")

        self.assertEqual(messages, [])
        self.assertEqual(log_file.read_text(encoding="utf-8"), "only to the file\n")

    def test_a_second_call_replaces_the_sink_of_the_first(self):
        first, second = self.tmp_path / "first.log", self.tmp_path / "second.log"
        configure_application_logging(first)
        configure_application_logging(second)

        logger.info("after reconfiguring")

        self.assertEqual(first.read_text(encoding="utf-8"), "")
        self.assertIn("after reconfiguring", second.read_text(encoding="utf-8"))

    def test_braces_in_the_file_name_are_kept(self):
        log_file = self.tmp_path / "{run}" / "app.log"
        configure_application_logging(log_file, format="{message}")

        logger.info("in the braces directory")

        self.assertEqual(
            log_file.read_text(encoding="utf-8"), "in the braces directory\n"
        )

    def test_nothing_is_written_without_stderr_and_without_a_file(self):
        messages: list[str] = []
        logger.add(messages.append, format="{message}")

        # sys.stderr is None in applications without a console.
        with mock.patch.object(sys, "stderr", None):
            configure_application_logging()
        logger.info("nowhere")

        self.assertEqual(messages, [])

    def test_stderr_sink_replaces_the_default_one(self):
        # A fresh interpreter, where Loguru has its default stderr sink, without
        # the Loguru settings of the environment that runs the tests.
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("LOGURU_")
        }
        environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        script = (
            "from loguru import logger\n"
            "from iree.core.log_config import configure_application_logging\n"
            "configure_application_logging(format='{message}')\n"
            "logger.info('exactly once')\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )

        self.assertEqual(result.stderr, "exactly once\n")


if __name__ == "__main__":
    unittest.main()
