#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Top level configure script conditionally calling more specific configuration
scripts"""

import pathlib
import shlex
import subprocess

from shutil import which


def run_shell(cmd_str, stdout=None):
  cmd = shlex.split(cmd_str)
  cmd_rslt = subprocess.run(cmd, check=True, text=True, stdout=stdout)

  if cmd_rslt.returncode != 0:
    raise RuntimeError(f"[ERROR] Command `{cmd_str}` returned with code: "
                       f"{cmd_rslt.returncode}")

  if stdout is not None:
    return cmd_rslt.stdout


def _get_input(question, default_answer):
  try:
    answer = input(f"\n- {question} ")
  except EOFError:
    answer = default_answer

  return (answer or default_answer).strip().lower()


def get_input(question, default_answer='', accepted_answers=None):
  if accepted_answers is None:
    raise RuntimeError("Argument `accepted_answers` is None.")

  accepted_answers = [x.strip().lower() for x in accepted_answers]

  while True:
    answer = _get_input(question, default_answer)
    if answer not in accepted_answers:
      print(f"\tERROR: Unsupported answer received: {answer}."
            f"Expected: {accepted_answers}")
      continue
    break

  return answer


def format_description(title, description):
  # Checking the title length.
  if len(title) > 69:
    raise ValueError(f"Title must be of length <= 69, received: {title}")
  title += " ..."

  # Trim any leading or trailing whitespace
  description = description.strip()

  # Split the description into chunks of up to 76 characters each
  chunks = []
  current_chunk = ""

  for word in description.split():
    if len(current_chunk) + len(word) + 1 > 76:
      chunks.append(current_chunk)
      current_chunk = ""
    current_chunk += f" {word}" if current_chunk else word

  if current_chunk:
    chunks.append(current_chunk)

  # Build the final formatted string
  formatted_string = ""
  for chunk in chunks:
    formatted_string += "| " + chunk.ljust(76) + " |\n"

  separator = "-" * 80
  return (f"\n{separator}\n| [*] {title.ljust(73)}|\n"
          f"|{''.ljust(78)}|\n{formatted_string}{separator}")


if __name__ == "__main__":

  IREE_DIR = pathlib.Path(__file__).resolve().parent

  # ====================== VSCode Development Container ===================== #
  print(
      format_description(
          title="VS Code Development Container",
          description="Dev containers let you work with a well-defined tool and "
          "runtime stack. See https://code.visualstudio.com/docs/devcontainers/containers."
      ))

  if get_input(
      "Do you wish to use and configure VS Code Development Container [y/N]?",
      default_answer="n",
      accepted_answers=["y", "n"]) == "y":

    run_shell(f"{which('python3')} {IREE_DIR / '.devcontainer/configure.py'}")

  # ================================= Bazel ================================= #
  print(
      format_description(
          title="Bazel",
          description=
          "While CMake is IREE's preferred build system, Bazel is also "
          "supported."))

  if get_input("Do you wish to use and configure Bazel [y/N]?",
               default_answer="n",
               accepted_answers=["y", "n"]) == "y":

    run_shell(
        f"{which('python3')} {IREE_DIR / 'build_tools/bazel/configure_bazel.py'}"
    )

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

  print("\n" + "-" * 80)
  print(f"| [*] {'Configuration finished ...'.ljust(72)} |")
  print("-" * 80)
