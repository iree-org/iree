#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import logging
import os
import subprocess
import unittest

NOTEBOOKS_TO_SKIP = [
    # matplotlib error when testing:
    #   FileNotFoundError: [Errno 2] No such file or directory: 'seaborn-whitegrid'
    # support level for TF-code and samples is also low
    "tensorflow_mnist_training.ipynb",
    # This needs 'transformers' (and possibly other packages) preinstalled.
    # Add to run_python_notebook.sh, colab/requirements.txt, or samples.yml?
    "pytorch_huggingface_whisper.ipynb",
]

NOTEBOOKS_EXPECTED_TO_FAIL = [
    # See https://github.com/iree-org/iree/issues/18919
    "tflite_text_classification.ipynb",
]


class ColabNotebookTests(unittest.TestCase):
    """Tests running all Colab notebooks in this directory."""

    @classmethod
    def generateTests(cls):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        script_path = os.path.join(
            repo_root, "build_tools/testing/run_python_notebook.sh"
        )

        # Create a test case for each notebook in this folder.
        notebooks_path = os.path.join(repo_root, "samples/colab/")
        for notebook_path in glob.glob(notebooks_path + "*.ipynb"):
            notebook_name = os.path.basename(notebook_path)

            def unit_test(self, notebook_path=notebook_path):
                completed_process = subprocess.run([script_path, notebook_path])
                self.assertEqual(completed_process.returncode, 0)

            if notebook_name in NOTEBOOKS_TO_SKIP:
                unit_test = unittest.skip("Skip requested")(unit_test)
            elif notebook_name in NOTEBOOKS_EXPECTED_TO_FAIL:
                unit_test = unittest.expectedFailure(unit_test)

            # Add 'unit_test' to this class, so the test runner runs it.
            unit_test.__name__ = f"test_{notebook_name}"
            setattr(cls, unit_test.__name__, unit_test)


if __name__ == "__main__":
    ColabNotebookTests.generateTests()
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
