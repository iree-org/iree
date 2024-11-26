# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import tempfile
import unittest

from iree.build import *
from iree.build.executor import BuildContext
from iree.build.test_actions import ExecuteOutOfProcessThunkAction


@entrypoint
def write_out_of_process_pid():
    context = BuildContext.current()
    output_file = context.allocate_file("pid.txt")
    action = ExecuteOutOfProcessThunkAction(
        _write_pid_file,
        args=[output_file.get_fs_path()],
        desc="Writing pid file",
        executor=context.executor,
    )
    output_file.deps.add(action)
    return output_file


def _write_pid_file(output_path: Path):
    pid = os.getpid()
    print(f"Running action out of process: pid={pid}")
    output_path.write_text(str(pid))


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._temp_dir.__enter__()
        self.output_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.__exit__(None, None, None)

    def testProcessConcurrency(self):
        parent_pid = os.getpid()
        print(f"Testing out of process concurrency: pid={parent_pid}")
        iree_build_main(
            args=["write_out_of_process_pid", "--output-dir", str(self.output_path)]
        )
        pid_file = (
            self.output_path / "genfiles" / "write_out_of_process_pid" / "pid.txt"
        )
        child_pid = int(pid_file.read_text())
        print(f"Got child pid={child_pid}")
        self.assertNotEqual(parent_pid, child_pid)


if __name__ == "__main__":
    unittest.main()
