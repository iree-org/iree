#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Potentially triggers and then waits for a Buildkite pipeline.

Checks if the given pipeline is already running for the given commit, starts
it if not, and then waits for it to complete. Exits successfully if the
triggered (or pre-existing) build succeeds, otherwise fails.
"""

import argparse
import os
from pybuildkite import buildkite
import sys
import time

# Fake build to return when running locally.
FAKE_PASSED_BUILD = dict(number=42, state="passed")


def get_build_number(build):
  return build.get("number")


class BuildkitePipelineManager():

  def __init__(
      self,
      *,
      access_token,
      pipeline,
      organization,
      commit,
      branch,
      message=None,
      author_name,
      author_email,
      pull_request_base_branch=None,
      pull_request_id=None,
      pull_request_repository=None,
  ):
    self._buildkite = buildkite.Buildkite()
    self._buildkite.set_access_token(access_token)
    self._pipeline = pipeline
    self._organization = organization
    self._commit = commit
    self._message = message
    self._branch = branch
    self._author_name = author_name
    self._author_email = author_email
    self._author = {"name": self._author_name, "email": self._author_email}
    self._pull_request_base_branch = pull_request_base_branch
    self._pull_request_id = pull_request_id
    self._pull_request_repository = pull_request_repository

  @staticmethod
  def from_environ(pipeline):
    # A token for the Buildkite API. Needs read/write privileges on builds to
    # watch and create builds. Within our pipelines we fetch this from secret
    # manager: https://cloud.google.com/secret-manager. Users can create a
    # personal token for running this script locally:
    # https://buildkite.com/docs/apis/managing-api-tokens
    access_token = os.environ["BUILDKITE_ACCESS_TOKEN"]

    # Buildkite sets these environment variables. See
    # https://buildkite.com/docs/pipelines/environment-variables. If running
    # locally you can set locally or use the simulate_buildkite.sh script.
    organization = os.environ["BUILDKITE_ORGANIZATION_SLUG"]
    commit = os.environ["BUILDKITE_COMMIT"]
    branch = os.environ["BUILDKITE_BRANCH"]

    # These variables aren't strictly necessary. Just nice to have (and set by
    # Buildkite).
    author_name = os.environ.get("BUILDKITE_BUILD_AUTHOR")
    author_email = os.environ.get("BUILDKITE_BUILD_AUTHOR_EMAIL")
    message = os.environ.get("BUILDKITE_MESSAGE")
    # These may not be set if build is not from a pull request
    pull_request_id = os.environ.get("BUILDKITE_PULL_REQUEST")
    pull_request_base_branch = os.environ.get(
        "BUILDKITE_PULL_REQUEST_BASE_BRANCH")
    pull_request_repository = os.environ.get("BUILDKITE_PULL_REQUEST_REPO")

    return BuildkitePipelineManager(
        access_token=access_token,
        organization=organization,
        pipeline=pipeline,
        commit=commit,
        message=message,
        branch=branch,
        author_name=author_name,
        author_email=author_email,
        pull_request_id=pull_request_id,
        pull_request_base_branch=pull_request_base_branch,
        pull_request_repository=pull_request_repository,
    )

  def get_builds(self):
    # Avoid API calls when running locally. The local organization doesn't
    # exist.
    if self._organization == "local":
      return [FAKE_PASSED_BUILD]
    return self._buildkite.builds().list_all_for_pipeline(
        organization=self._organization,
        pipeline=self._pipeline,
        commit=self._commit)

  def get_build_by_number(self, build_number):
    # Avoid API calls when running locally. The local organization doesn't
    # exist.
    if (self._organization == "local" and
        build_number == get_build_number(FAKE_PASSED_BUILD)):
      print("Returning fake build because running locally")
      return FAKE_PASSED_BUILD
    return self._buildkite.builds().get_build_by_number(self._organization,
                                                        self._pipeline,
                                                        build_number)

  def get_latest_build(self):
    all_builds = self.get_builds()
    if not all_builds:
      return None
    return max(all_builds, key=get_build_number)

  def create_build(self):
    return self._buildkite.builds().create_build(
        organization=self._organization,
        pipeline=self._pipeline,
        commit=self._commit,
        message=self._message,
        branch=self._branch,
        ignore_pipeline_branch_filters=True,
        author=self._author,
        pull_request_base_branch=self._pull_request_base_branch,
        pull_request_id=self._pull_request_id,
        pull_request_repository=self._pull_request_repository,
    )

  def wait_for_build(self, build_number):
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
      build = self.get_build_by_number(build_number)
      state = buildkite.BuildState(build["state"])
      wait_time = int(round(time.monotonic() - start))
      if state in [
          buildkite.BuildState.PASSED,
          buildkite.BuildState.FAILED,
          buildkite.BuildState.CANCELED,
          buildkite.BuildState.SKIPPED,
          buildkite.BuildState.NOT_RUN,
      ]:
        output_str = f"Build finished in state '{state.name}'"
        min_line_length = max(min_line_length, len(output_str))
        print(output_str.ljust(min_line_length))
        return state

      output_str = (
          f"Waiting for build {build_number} to complete. Waited {wait_time}"
          f" seconds. Currently in state '{state.name}':"
          f" {linkify(self.get_url_for_build(build_number))}")
      min_line_length = max(min_line_length, len(output_str))
      print(output_str.ljust(min_line_length), "\r", end="", flush=True)
      # Yes, polling is unfortunately the best we can do here :-(
      time.sleep(5)

  def get_url_for_build(self, build_number):
    return f"https://buildkite.com/{self._organization}/{self._pipeline}/builds/{build_number}"


# Make a link clickable using ANSI escape sequences. See
# https://buildkite.com/docs/pipelines/links-and-images-in-log-output
def linkify(url, text=None):
  if text is None:
    text = url

  return f"\033]1339;url={url};content={text}\a"


def should_create_new_build(bk, build, rebuild_option):
  if not build:
    print("Didn't find previous build for pipeline. Creating a new one.")
    return True
  state = buildkite.BuildState(build["state"])
  build_number = get_build_number(build)
  url = bk.get_url_for_build(build_number)
  print(f"Found previous build with state '{state.name}': {url}")

  if rebuild_option == "force":
    print(f"Received `--rebuild=force`, so creating a new build")
    return True
  elif rebuild_option == "failed" and state == buildkite.BuildState.FAILED:
    print(f"Previous build failed and received `--rebuild=failed`, so"
          f" creating a new one.")
    return True
  elif rebuild_option == "bad" and state in (
      buildkite.BuildState.FAILED,
      buildkite.BuildState.CANCELED,
      buildkite.BuildState.SKIPPED,
      buildkite.BuildState.NOT_RUN,
  ):
    print(f"Previous build completed with state '{state.name}' and received"
          f" `--rebuild=bad`, so creating a new one.")
    return True

  return False


def parse_args():
  parser = argparse.ArgumentParser(
      description="Waits on the status of the last Buildkite build for a given"
      " commit or creates such a build if none exists")
  parser.add_argument(
      "pipeline", help="The pipeline for which to create and wait for builds")
  parser.add_argument(
      "--rebuild",
      help="Behavior for triggering a new build even if there is an existing"
      " one. `force`: always rebuild without checking for existing build,"
      " `failed`: rebuild on build finished in 'failed' state, `bad`: rebuild"
      " on build finished in state other than 'passed'",
      choices=["force", "failed", "bad"],
  )
  return parser.parse_args()


def main(args):
  bk = BuildkitePipelineManager.from_environ(args.pipeline)
  build = bk.get_latest_build()

  if should_create_new_build(bk, build, args.rebuild):
    build = bk.create_build()
  build_number = get_build_number(build)
  url = bk.get_url_for_build(build_number)
  print(f"Waiting on {linkify(url)}")
  state = bk.wait_for_build(build_number)
  if state != buildkite.BuildState.PASSED:
    print(f"Build was not successful: {linkify(url)}")
    sys.exit(1)
  print(f"Build completed successfully: {linkify(url)}")


if __name__ == "__main__":
  main(parse_args())
