# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import os
from pathlib import Path
import tempfile
import unittest

from iree.build import *
from iree.build.executor import BuildContext
from iree.build.test_actions import ExecuteOutOfProcessThunkAction


TEST_URL = None
TEST_URL_1 = "https://huggingface.co/google-bert/bert-base-cased/resolve/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer.json"
TEST_URL_2 = "https://huggingface.co/google-bert/bert-base-cased/resolve/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer_config.json"


@entrypoint
def tokenizer_via_http():
    return fetch_http(
        name="tokenizer.json",
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
        path = self.output_path / "genfiles" / "tokenizer_via_http" / "tokenizer.json"

        def run():
            nonlocal out
            nonlocal err
            try:
                out_io = io.StringIO()
                err_io = io.StringIO()
                iree_build_main(
                    args=[
                        "tokenizer_via_http",
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
