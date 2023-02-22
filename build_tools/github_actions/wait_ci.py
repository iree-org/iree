#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TODO"""

import argparse
import http.client
import os
import pathlib
import requests
import sys
import time
from typing import Any, Dict

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/openxla/iree"
GITHUB_API_VERSION = "2022-11-28"


def _wait_for_job(session: requests.Session,
                  github_token: str,
                  commit_sha: str,
                  job_name: str,
                  retry_wait_time_sec: int = 30,
                  max_wait_time_sec: int = 6 * 60 * 60) -> Dict[str, Any]:
  """Wait for a job run on a commit.

  Args:
    session: requests session.
    github_token: Github token.
    commit_sha: commit SHA.
    job_name: job name to wait.
    retry_wait_time_sec: Time to wait before retry, in seconds.
    max_wait_time_sec: Total time to wait before timeout, in seconds.

  Returns:
    Github job run object.
  """
  start_time = time.time()
  found_job_run = None
  while (time.time() - start_time) < max_wait_time_sec:
    resp = session.get(
        f"{GITHUB_IREE_API_PREFIX}/commits/{commit_sha}/check-runs",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {github_token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        params={"check_name": job_name})
    if not resp.ok:
      raise RuntimeError(
          f"Failed to fetch check runs; error code: {resp.status_code} - {resp.text}"
      )

    job_runs = [
        run for run in resp.json()["check_runs"] if run["name"] == job_name
    ]
    job_run_count = len(job_runs)
    if job_run_count > 1:
      raise RuntimeError(f"Found {job_run_count} instead of 1 {job_name} job.")
    elif job_run_count == 1:
      job_run = job_runs[0]
      status = job_run["status"]
      if status == "completed":
        found_job_run = job_run
        break
      elif status not in ["queued", "in_progress"]:
        raise ValueError(f"Unexpected job status: {status}")

    print(
        f"Waiting for {job_name}... {time.time() - start_time:.0f}s have passed.",
        flush=True)
    time.sleep(retry_wait_time_sec)

  if found_job_run is None:
    raise TimeoutError(f"Timeout on waiting for {job_name}")

  return found_job_run


def _parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("job_name", type=str)
  parser.add_argument("--output_variables", type=pathlib.Path, default=None)
  parser.add_argument("--retry_wait_time_sec", type=int, default=30)
  parser.add_argument("--max_wait_time_sec", type=int, default=6 * 60 * 60)

  return parser.parse_args()


def main(args: argparse.Namespace):
  github_token = os.environ["GITHUB_TOKEN"]
  commit_sha = os.environ["HEAD_SHA"]
  job_name = args.job_name

  session = requests.session()
  found_job_run = _wait_for_job(session=session,
                                github_token=github_token,
                                commit_sha=commit_sha,
                                job_name=job_name,
                                retry_wait_time_sec=args.retry_wait_time_sec,
                                max_wait_time_sec=args.max_wait_time_sec)

  conclusion = found_job_run["conclusion"]
  outputs = [f"wait-conclusion={conclusion}"]
  if conclusion == "failure":
    print(f"{job_name} failed.")
  elif conclusion == "skipped":
    print(f"{job_name} is skipped.")
  elif conclusion == "success":
    resp = session.get(
        found_job_run["output"]["annotations_url"],
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {github_token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    if not resp.ok:
      raise RuntimeError(
          f"Failed to fetch annotations; error code: {resp.status_code} - {resp.text}"
      )
    output_messages = [
        annotation["message"]
        for annotation in resp.json()
        if annotation["title"] == "JOB_OUTPUT_VARIABLE"
    ]
    print(f"{job_name} succeeded, outputs variables:")
    print("\n".join(output_messages))
    outputs.extend(output_messages)
  else:
    raise ValueError(f"Unexpected job conclusion: {conclusion}")

  if args.output_variables is not None:
    args.output_variables.write_text("\n".join(outputs) + "\n")


if __name__ == "__main__":
  main(_parse_arguments())
