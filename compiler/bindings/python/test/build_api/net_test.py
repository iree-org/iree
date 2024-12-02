# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from pathlib import Path
import tempfile
import unittest

from iree.build import *


TEST_URL = None
# Arbitrary URLs to download from via HTTP requests. These should require no
# authentication to access and should ideally sit behind a CDN that can handle
# random CI and developer traffic. We could also mock the fetching to make the
# tests hermetic.
TEST_URL_1 = "https://raw.githubusercontent.com/iree-org/iree/82724905d64eebb2f62bcc0e41626a7b5156fd8f/.gitignore"
TEST_URL_2 = "https://raw.githubusercontent.com/iree-org/iree/82724905d64eebb2f62bcc0e41626a7b5156fd8f/.gitmodules"


@entrypoint
def file_via_http():
    return fetch_http(
        name="file.txt",
        url=TEST_URL,
    )


class BasicTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._temp_dir.__enter__()
        self.output_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.__exit__(None, None, None)

    def test_fetch_http(self):
        # This just does a sanity check that rich console mode does not crash. Actual
        # behavior can really only be completely verified visually.
        out = None
        err = None
        global TEST_URL
        path = self.output_path / "genfiles" / "file_via_http" / "file.txt"

        def run():
            nonlocal out
            nonlocal err
            try:
                out_io = io.StringIO()
                err_io = io.StringIO()
                iree_build_main(
                    args=[
                        "file_via_http",
                        "--output-dir",
                        str(self.output_path),
                        "--test-force-console",
                    ],
                    stderr=err_io,
                    stdout=out_io,
                )
            finally:
                out = out_io.getvalue()
                err = err_io.getvalue()
                print(f"::test_fetch_http err: {err!r}")
                print(f"::test_fetch_http out: {out!r}")

        def assertExists():
            self.assertTrue(path.exists(), msg=f"Path {path} exists")

        # First run should fetch.
        TEST_URL = TEST_URL_1
        run()
        self.assertIn("Fetching URL: https://", err)
        assertExists()

        # Second run should not fetch.
        TEST_URL = TEST_URL_1
        run()
        self.assertNotIn("Fetching URL: https://", err)
        assertExists()

        # Fetching a different URL should download again.
        TEST_URL = TEST_URL_2
        run()
        self.assertIn("Fetching URL: https://", err)
        assertExists()

        # Removing the file should fetch again.
        TEST_URL = TEST_URL_2
        path.unlink()
        run()
        self.assertIn("Fetching URL: https://", err)
        assertExists()


if __name__ == "__main__":
    unittest.main()
