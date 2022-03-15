#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks if the given pipeline is already running for the given commit, starts
# it if not, and then waits for it to complete.

import argparse
import os
from pybuildkite import buildkite
import sys
import time


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
    organization = os.environ["BUILDKITE_ORGANIZATION_SLUG"]
    # This is not one of Buildkite's. We set it ourself by fetching the secret
    # from secret manager (a human can set this to their own personal access
    # token)
    access_token = os.environ["BUILDKITE_ACCESS_TOKEN"]
    commit = os.environ["BUILDKITE_COMMIT"]
    branch = os.environ["BUILDKITE_BRANCH"]
    author_name = os.environ["BUILDKITE_BUILD_AUTHOR"]
    author_email = os.environ["BUILDKITE_BUILD_AUTHOR_EMAIL"]
    # If this isn't set (most likely in a local run) Buildkite will populate it
    # based on the commit, so it's not a problem if they're not set. It's still
    # nice to set it if we can because Buildkite can only set it once the build
    # is accepted by an agent and the git repository is checked out, so the
    # build will otherwise be nameless until that point.
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
    return self._buildkite.builds().list_all_for_pipeline(
        organization=self._organization,
        pipeline=self._pipeline,
        commit=self._commit)

  def get_build_by_number(self, build_number):
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
          f" {self.get_url_for_build(build_number)}")
      min_line_length = max(min_line_length, len(output_str))
      print(output_str.ljust(min_line_length), "\r", end="", flush=True)
      # Yes, polling is unfortunately the best we can do here :-(
      time.sleep(5)

  def get_url_for_build(self, build_number):
    return f"https://buildkite.com/{self._organization}/{self._pipeline}/builds/{build_number}"


def parse_args():
  parser = argparse.ArgumentParser(
      description="Waits on the status of the last BuildKite build for a given"
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
  create_new_build = True

  if build:
    create_new_build = False
    state = buildkite.BuildState(build["state"])
    build_number = get_build_number(build)
    url = bk.get_url_for_build(build_number)
    print(f"Found previous build with state '{state.name}': {url}")

    if args.rebuild == "force":
      print(f"Received `--rebuild=force`, so creating a new build")
      create_new_build = True
    elif args.rebuild == "failed" and state == buildkite.BuildState.FAILED:
      print(f"Previous build failed and received `--rebuild=failed`, so"
            f" creating a new one.")
      create_new_build = True
    elif args.rebuild == "bad" and state in (
        buildkite.BuildState.FAILED,
        buildkite.BuildState.CANCELED,
        buildkite.BuildState.SKIPPED,
        buildkite.BuildState.NOT_RUN,
    ):
      print(f"Previous build completed with state '{state.name}' and received"
            f" `--rebuild=bad`, so creating a new one.")
      create_new_build = True
  else:
    print("Didn't find previous build for pipeline. Creating a new one.")

  if create_new_build:
    build = bk.create_build()
  build_number = get_build_number(build)
  state = bk.wait_for_build(build_number)

  build_number = get_build_number(build)
  url = bk.get_url_for_build(build_number)
  if state != buildkite.BuildState.PASSED:
    print(f"Build was not successful: {url}")
    sys.exit(1)
  print(f"Build completed successfully: {url}")


if __name__ == "__main__":
  main(parse_args())
