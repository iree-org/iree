# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import pytest
import shutil

ARTIFACT_DIR_KEY = pytest.StashKey[Path]()
TEST_DIR_KEY = pytest.StashKey[Path]()
MAX_PATHLENGTH = os.pathconf('/', 'PC_NAME_MAX')


def pytest_addoption(parser, pluginmanager) -> None:
    parser.addoption(
        "--openxla-pjrt-artifact-dir",
        dest="OPENXLA_PJRT_ARTIFACT_DIR",
        help="Saves OpenXLA PJRT compilation artifacts",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    artifact_dir = session.config.getoption("OPENXLA_PJRT_ARTIFACT_DIR")
    session.stash[ARTIFACT_DIR_KEY] = Path(artifact_dir)


def pytest_runtest_setup(item: pytest.Item) -> None:
    artifact_dir = item.session.stash[ARTIFACT_DIR_KEY]
    if artifact_dir is None:
        return
    sanitized_name = (
        item.nodeid.replace(".py::", "::").replace("/", "_").replace("::", "__")
    )

    test_dir = artifact_dir / sanitized_name
    if len(sanitized_name) > MAX_PATHLENGTH:
        test_dir = artifact_dir / str(hash(sanitized_name))
    else:
        test_dir = artifact_dir / sanitized_name

    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    item.stash[TEST_DIR_KEY] = test_dir
    os.putenv("IREE_PJRT_SAVE_ARTIFACTS", str(test_dir))
    with open(test_dir / "NAME", "wt") as f:
        f.write(sanitized_name)

    with open(test_dir / "CRASH_MARKER", "wt") as f:
        f.write("If this file exists, the test crashed or was killed")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call) -> None:
    outcome = yield
    test_dir = item.stash[TEST_DIR_KEY]
    if test_dir is None:
        return

    result = outcome.get_result()

    if call.when == "call" and result.failed:
        with open(test_dir / "error.txt", "wt") as f:
            f.write(result.longreprtext)
            f.write("\n\nSTDERR:\n-------\n")
            f.write(result.capstderr)
            f.write("\n\nLOG:\n----\n")
            f.write(result.caplog)
            f.write("\n\nSTDOUT:\n-------\n")
            f.write(result.capstdout)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    test_dir = item.stash[TEST_DIR_KEY]
    if test_dir is None:
        return
    dir_entries = list(test_dir.iterdir())
    crash_marker = test_dir / "CRASH_MARKER"
    if crash_marker.is_file():
        crash_marker.unlink()
    if not dir_entries:
        # Remove empty directories on success.
        test_dir.rmdir()
