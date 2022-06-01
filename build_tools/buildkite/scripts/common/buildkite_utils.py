#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Dict, Optional

import requests
from pybuildkite import buildkite

# Type signature of Buildkite build object.
BuildObject = Dict[str, Any]


def get_build_number(build: BuildObject) -> int:
  return build["number"]


def get_build_state(build: BuildObject) -> buildkite.BuildState:
  return buildkite.BuildState(build.get("state"))


def linkify(url: str, text: Optional[str] = None):
  """Make a link clickable using ANSI escape sequences.
    See https://buildkite.com/docs/pipelines/links-and-images-in-log-output
  """

  if text is None:
    text = url

  return f"\033]1339;url={url};content={text}\a"


def get_pipeline(bk, *, organization, pipeline_slug):
  try:
    pipeline = bk.pipelines().get_pipeline(organization, pipeline_slug)
  except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
      return None
    raise e
  return pipeline
