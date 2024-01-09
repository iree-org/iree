#!/usr/bin/env python3

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import subprocess
import os

# This is a very simple test runner for tests that require Arm SME.
# This assumes the host is an aarch64 machine.

if __name__ == "__main__":
    build_dir = os.getenv("BUILD_DIR", ".")
    os.chdir(build_dir)

    print("*************** Running ArmSME Tests ***************")

    qemu_aarch64_executable = os.getenv("QEMU_AARCH64_EXECUTABLE", "qemu-aarch64")

    # Gather list of tests that use ArmSME.
    arm_sme_test_info = json.loads(
        subprocess.check_output(
            ["ctest", "--label-regex", "^requires-arm-sme$", "--show-only=json-v1"]
        )
    )

    for test in arm_sme_test_info["tests"]:
        name = test["name"]
        test_command = [
            qemu_aarch64_executable,
            "-cpu",
            "max,sme512=on,sme_fa64=off",
            "--",
            *test["command"],
        ]

        # Ensure the test YAML/VFMB are built.
        # Note: This is allowed to fail, as if this target is needed and it fails
        # to build, the `test_command` should fail.
        target_name = name.replace("/", "_")
        subprocess.run(["ninja", target_name])

        print()
        print("Running ArmSME test:", name)
        print("Test command:", test_command)

        # Run and check the result of running the test command under QEMU:
        subprocess.run(test_command, check=True)
