#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ghapi.all import GhApi
import argparse
import dataclasses
import os
import sys
import time

CLA_CHECK_NAME = "cla/google"
REPO_OWNER = "google"
REPO_NAME = "iree"


class ApiWrapper:

  def __init__(self):
    self._api = GhApi()

  def get_cla_check(self, ref):
    response = self._api.checks.list_for_ref(owner=REPO_OWNER,
                                             repo=REPO_NAME,
                                             ref=ref,
                                             check_name=CLA_CHECK_NAME)
    if len(response.check_runs) == 1:
      return response.check_runs[0]
    elif len(response.check_runs) == 0:
      return None

  def wait_for_cla_check(self, ref, wait_secs=2):
    # We want to override the previous output when logging about waiting, so
    # this doesn't print a bunch of unhelpful log lines. Carriage return takes
    # us back to the beginning of the line, but it doesn't override previous
    # output past the end of the new output, so we pad things to ensure that
    # each line is at least as long as the previous one. Note that this approach
    # only works if a print statement doesn't overflow a single line (at least
    # on my machine). In that case, the beginning of the line is partway through
    # the previous print, although it at least starts on a new line. Better
    # suggestions welcome.
    min_line_length = 0
    # We don't need great precision
    start = time.monotonic()
    while True:
      cla_check = self.get_cla_check(ref)
      if cla_check is not None and cla_check.status == "completed":
        return cla_check

      wait_time = int(round(time.monotonic() - start))
      output_str = (
          f"Waiting for CLA Check to complete on {ref}. Waited {wait_time}"
          f" seconds."
          f" CLA Check is currently"
          f" {'not found' if cla_check is None else cla_check.status}")
      min_line_length = max(min_line_length, len(output_str))
      print(output_str.ljust(min_line_length), "\r", end="", flush=True)

      time.sleep(wait_secs)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Check if the given commit has passed the Google CLA check")
  parser.add_argument("ref", help="Git reference to check.")
  parser.add_argument(
      "--wait",
      nargs="?",
      metavar="N",
      help="Poll for a complete check every N seconds if one is not found."
      " Without this flag, the script will only check once. If this is set"
      " without a value, if will check every 2 seconds.",
      default=0,
      const=2,
  )
  return parser.parse_args()


def main(args):
  wrapper = ApiWrapper()
  if args.wait:
    cla_check = wrapper.wait_for_cla_check(args.ref, wait_secs=args.wait)
  else:
    cla_check = wrapper.get_cla_check(args.ref)

  # Some of these should be impossible for the wait case, but it's simpler to
  # just check them again.

  if cla_check is None:
    print(f"Didn't find CLA check for '{args.ref}'")
    sys.exit(3)

  if cla_check.status != "completed":
    print(f"CLA check has not finished running for '{args.ref}'."
          f" Is '{cla_check.status}'")
    sys.exit(2)

  # Doing it this way round so if there is some mistake in this check (e.g.
  # a typo, or 'success' is changed to 'passed') the whole thing fails.
  if cla_check.conclusion == "success":
    print(f"CLA check passed for '{args.ref}'")
    sys.exit(0)

  print(f"CLA check did not succeed for '{args.ref}'."
        f" Completed with '{cla_check.conclusion}'")
  sys.exit(1)


if __name__ == "__main__":
  main(parse_args())
