#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import time
from typing import List, Optional

from pybuildkite import buildkite

from common.buildkite_utils import (BuildObject, get_build_number,
                                    get_build_state, get_pipeline, linkify)

# Fake build to return when running locally.
FAKE_PASSED_BUILD = dict(number=42, state="passed")

# The special pipeline that runs unregistered pipelines
UNREGISTERED_PIPELINE_NAME = "unregistered"


class BuildkitePipelineManager(object):
  """Buildkite pipeline manager."""

  def __init__(
      self,
      access_token: str,
      pipeline: str,
      organization: str,
      commit: str,
      branch: str,
      author_name: Optional[str] = None,
      author_email: Optional[str] = None,
      message: Optional[str] = None,
      pull_request_base_branch: Optional[str] = None,
      pull_request_id: Optional[str] = None,
      pull_request_repository: Optional[str] = None,
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
    self._env = {
        # Indicate the name of the pipeline to trigger. This is used by the
        # 'unregistered' pipeline to upload the correct pipeline file.
        "REQUESTED_PIPELINE": pipeline,
    }
    # If the pipeline doesn't exist, then we run it on the "unregistered"
    # pipeline, which leverages the REQUESTED_PIPELINE env variable.
    if not get_pipeline(self._buildkite,
                        organization=self._organization,
                        pipeline_slug=self._pipeline):
      self._pipeline = UNREGISTERED_PIPELINE_NAME

  @staticmethod
  def from_environ(pipeline: str):
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

  def get_builds(self) -> List[BuildObject]:
    # Avoid API calls when running locally. The local organization doesn't
    # exist.
    if self._organization == "local":
      return [FAKE_PASSED_BUILD]
    return self._buildkite.builds().list_all_for_pipeline(
        organization=self._organization,
        pipeline=self._pipeline,
        commit=self._commit)

  def get_build_by_number(self, build_number: int) -> BuildObject:
    # Avoid API calls when running locally. The local organization doesn't
    # exist.
    if (self._organization == "local" and
        build_number == get_build_number(FAKE_PASSED_BUILD)):
      print("Returning fake build because running locally")
      return FAKE_PASSED_BUILD
    return self._buildkite.builds().get_build_by_number(self._organization,
                                                        self._pipeline,
                                                        build_number)

  def get_latest_build(self) -> Optional[BuildObject]:
    # Builds of the unregistered pipeline are not necessarily performing the
    # same work.
    if self._pipeline == UNREGISTERED_PIPELINE_NAME:
      return None
    all_builds = self.get_builds()
    if not all_builds:
      return None
    return max(all_builds, key=get_build_number)

  def create_build(self) -> BuildObject:
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
        env=self._env,
    )

  def wait_for_build(self, build_number: int) -> BuildObject:
    # We want to override the previous output when logging about waiting, so
    # this doesn't print a bunch of unhelpful log lines. Carriage return takes
    # us back to the beginning of the line, but it doesn't override previous
    # output past the end of the new output, so we pad things to ensure that
    # each line is at least as long as the previous one. Note that this approach
    # only works if a print statement doesn't overflow a single line (at least
    # on my machine). In that case, the beginning of the line is partway through
    # the previous print, although it at least starts on a new line. Better
    # suggestions welcome.
    max_line_length = 0
    # We don't need great precision
    start = time.monotonic()
    while True:
      build = self.get_build_by_number(build_number)
      state = get_build_state(build)
      if state in [
          buildkite.BuildState.PASSED,
          buildkite.BuildState.FAILED,
          buildkite.BuildState.CANCELED,
          buildkite.BuildState.SKIPPED,
          buildkite.BuildState.NOT_RUN,
      ]:
        output_str = f"Build finished in state '{state.name}'"
        max_line_length = max(max_line_length, len(output_str))
        print(output_str.ljust(max_line_length))
        return build

      wait_time = int(round(time.monotonic() - start))
      output_str = (
          f"Waiting for build {build_number} to complete. Waited {wait_time}"
          f" seconds. Currently in state '{state.name}':"
          f" {linkify(self.get_url_for_build(build_number))}")
      max_line_length = max(max_line_length, len(output_str))
      print(output_str.ljust(max_line_length), "\r", end="", flush=True)
      # Yes, polling is unfortunately the best we can do here :-(
      time.sleep(5)

  def get_url_for_build(self, build_number: int) -> str:
    return f"https://buildkite.com/{self._organization}/{self._pipeline}/builds/{build_number}"
