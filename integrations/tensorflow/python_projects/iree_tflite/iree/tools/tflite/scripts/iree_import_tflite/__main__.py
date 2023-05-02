# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import inspect
from pathlib import Path
import re
import sys
import iree.tools.tflite


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('flatbuffer', help='<TFLite FlatBuffer>')
  parser.add_argument(
      '-o',
      '--output-path',
      dest='output_path',
      required=True,
      help='Path to the mlirbc file name to output.',
  )
  parser.add_argument(
      '--input-array',
      dest='input_arrays',
      action='append',
      help='Input tensor, if different from the default inputs',
  )
  parser.add_argument(
      '--output-array',
      dest='output_arrays',
      action='append',
      help='Output tensor, if different from the default inputs',
  )
  args = parser.parse_args()
  tflite_to_tosa(
      flatbuffer=args.flatbuffer,
      bytecode=args.output_path,
      ordered_input_arrays=input_arrays,
      ordered_output_arrays=output_arrays,
  )


def tflite_to_tosa(
    flatbuffer,
    bytecode,
    use_external_constant=False,
    ordered_input_arrays=None,
    ordered_output_arrays=None,
):
  from tensorflow.python.pywrap_mlir import experimental_tflite_to_tosa_bytecode
  experimental_tflite_to_tosa_bytecode(flatbuffer, bytecode,
                                       use_external_constant,
                                       ordered_input_arrays,
                                       ordered_output_arrays)


if __name__ == "__main__":
  main()
