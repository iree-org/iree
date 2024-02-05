#!/bin/python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates the TFLite integration test model documentation.

When changes are made to any models in this directory, please run this script to
update the dashboard of models.
"""

import os
from pathlib import Path
from typing import Sequence

# The symbols to show in the table if the operation is supported or not.
SUCCESS_ELEMENT = "PASS ✓"
FAILURE_ELEMENT = "FAIL ✗"


def main():
    dir = os.path.dirname(__file__)
    readme_file_path = os.path.join(dir, "README.md")
    old_lines = read_file(readme_file_path)

    files = list(Path(dir).glob("*.run"))
    num_files = len(files)

    models = [[0 for x in range(2)] for y in range(num_files)]
    print(f"Processing {num_files} files")

    for i in range(num_files):
        name = os.path.basename(files[i].name).replace(".run", "")
        models[i][0] = name.ljust(20)

        with open(files[i], "r") as file:
            models[i][1] = (
                FAILURE_ELEMENT if "XFAIL" in file.read() else SUCCESS_ELEMENT
            )

    with open(readme_file_path, "w", encoding="utf-8") as tflite_model_documentation:
        tflite_model_documentation.write(
            "# TFLite integration tests status\n\n"
            "This dashboard shows the models that are currently being tested on IREE's\n"
            "presubmits.  If any tests are added or changed, please run\n"
            "update_tflite_model_documentation.py to update this table.\n\n"
            "|       Model        |      Status        |\n"
            "| ------------------ | ------------------ |\n"
        )
        tflite_model_documentation.write(create_markdown_table(models))

    new_lines = read_file(readme_file_path)
    if new_lines == old_lines:
        print(f"{readme_file_path} required no update")
    else:
        print(f"Updated {readme_file_path} with latest test status")


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.readlines()


def create_markdown_table(rows: Sequence[Sequence[str]]):
    """Converts a 2D array to a Markdown table."""
    return "\n".join([" | ".join(row) for row in rows])


if __name__ == "__main__":
    main()
