# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# MLIR dialect markdown document postprocessor for website generation.

import argparse
import os
import fileinput
import re


def main(args):
    directory = args.directory
    files = [os.path.join(directory, f) for f in os.listdir(directory)]

    with fileinput.input(files=files, inplace=True) as f:
        for line in f:
            # Replace certain headings with one level deeper.
            # Skipping heading levels is usually discouraged, but we aren't
            # getting much value from treating these as subsections and we
            # don't want them showing up in the rendered table of contents.
            # (We could use another form of emphasis like bolt/italics instead)
            line = re.sub(r"^#### Attributes", "##### Attributes", line)
            line = re.sub(r"^#### Parameters", "##### Parameters", line)
            line = re.sub(r"^#### Operands", "##### Operands", line)
            line = re.sub(r"^#### Results", "##### Results", line)

            print(line, end="")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dialect doc postprocessor.")

    parser.add_argument(
        "directory",
        help="Dialect docs directory to edit in-place.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
