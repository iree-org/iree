# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse

import iree.compiler.tf
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('saved_model_path',
                    help='Path to the saved model directory to import.')
parser.add_argument('-o',
                    '--output_path',
                    dest='output_path',
                    required=True,
                    help='Path to the mlir file name to output.')

# Deprecated and unused.  Kept in place so callers of the old tool don't break
# when using the new tool.
parser.add_argument('--tf-savedmodel-exported-names',
                    dest='_',
                    required=False,
                    help=argparse.SUPPRESS)
parser.add_argument('--tf-import-type',
                    dest='_',
                    required=False,
                    help=argparse.SUPPRESS)
parser.add_argument('--output-format',
                    dest='_',
                    required=False,
                    help=argparse.SUPPRESS)
args = parser.parse_args()


def main():
  Path(args.output_path).write_text(
      iree.compiler.tf.get_mlir(args.saved_model_path))


if __name__ == "__main__":
  main()
