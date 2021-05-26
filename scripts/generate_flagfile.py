#!/usr/bin/env python3

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates a flagfile for iree-benchmark-module."""

import argparse
import os


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--module_file",
                      type=str,
                      required=True,
                      metavar="<module-file>",
                      help="The name of the module file")
  parser.add_argument("--driver",
                      type=str,
                      required=True,
                      metavar="<driver>",
                      help="The name of IREE driver")
  parser.add_argument("--additional_args",
                      type=str,
                      required=True,
                      metavar="<additional-cl-args>",
                      help="Additional command-line arguments")
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      metavar="<output-file>",
                      help="Output file to write to")
  return parser.parse_args()


def main(args):
  lines = [f"--module_file={args.module_file}", f"--driver={args.driver}"]
  lines.extend(args.additional_args.split(";"))
  content = "\n".join(lines) + "\n"

  with open(args.output, "w") as f:
    f.writelines(content)


if __name__ == "__main__":
  main(parse_arguments())
