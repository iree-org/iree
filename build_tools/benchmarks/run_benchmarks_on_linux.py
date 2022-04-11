#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on a Linux device."""

from common.common_arguments import build_common_argument_parser
from common.linux_device_utils import get_linux_device_info


def main(args):
  device_info = get_linux_device_info(args.device_model, args.verbose)
  if args.verbose:
    print(device_info)

  raise NotImplementedError()


def parse_argument():
  arg_parser = build_common_argument_parser()
  arg_parser.add_argument("--device_model",
                          default="Unknown",
                          help="Device model")

  return arg_parser.parse_args()


if __name__ == "__main__":
  main(parse_argument())
