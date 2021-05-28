#!/usr/bin/env python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
from typing import Sequence

PROD_DIGESTS_PATH = "build_tools/docker/prod_digests.txt".replace("/", os.sep)


def run_command(command: Sequence[str],
                dry_run: bool = False,
                check: bool = True,
                capture_output: bool = False,
                text: bool = True,
                **run_kwargs) -> subprocess.CompletedProcess:
  """Thin wrapper around subprocess.run"""
  print(f"Running: `{' '.join(command)}`")
  if dry_run:
    # Dummy CompletedProess with successful returncode.
    return subprocess.CompletedProcess(command, returncode=0)

  if capture_output:
    # TODO(#4131) python>=3.7: Use capture_output=True.
    run_kwargs["stdout"] = subprocess.PIPE
    run_kwargs["stderr"] = subprocess.PIPE

  # TODO(#4131) python>=3.7: Replace 'universal_newlines' with 'text'.
  completed_process = subprocess.run(command,
                                     universal_newlines=text,
                                     check=check,
                                     **run_kwargs)
  return completed_process


def check_gcloud_auth(dry_run: bool = False):
  # Ensure the user has the correct authorization if they try to push to GCR.
  try:
    run_command(['which', 'gcloud'])
  except subprocess.CalledProcessError as error:
    raise RuntimeError(
        'gcloud not found. See https://cloud.google.com/sdk/install for '
        'installation.') from error
  run_command(["gcloud", "auth", "configure-docker"], dry_run)
