#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates all Buildkite pipelines with the Buildkite API.

This overwrites the configuration of all the Buildkite pipeline. Pipelines
should be updated this way, not in the UI. Before updating, this checks for a
specific header in the configuration stored in the API that indicates which
Buildkite build previously updated it. If the updater was not this same pipeline
the script fails. If it was a later build in this pipeline, it skips updating
unless the `--force` flag is passed. This is to avoid situations where an
earlier build is for some reason running a step after a later build due to race
conditions or a retry. Buildkite concurrency groups should be used to prevent
builds from trying to update pipelines simultaneously.
"""

import argparse
import glob
import os
import subprocess
import sys
import urllib

import requests
from pybuildkite import buildkite

PIPELINE_ROOT_PATH = "build_tools/buildkite/pipelines"
BUILD_URL_PREAMBLE = "# Automatically updated by Buildkite pipeline"

UPDATE_INFO_HEADER = f"""{BUILD_URL_PREAMBLE} {{build_url}}
# from pipeline file {{pipeline_file_url}}
# with script {{script_url}}

"""


def get_git_root():
  return subprocess.run(["git", "rev-parse", "--show-toplevel"],
                        check=True,
                        stdout=subprocess.PIPE,
                        text=True).stdout.strip()


def should_update(bk, *, organization, pipeline_file, running_pipeline,
                  running_build_number):
  pipeline_to_update, _ = os.path.splitext(os.path.basename(pipeline_file))
  previous_pipeline_configuration = bk.pipelines().get_pipeline(
      organization, pipeline_to_update)["configuration"]

  first_line, _ = previous_pipeline_configuration.split("\n", 1)
  if not first_line.startswith(BUILD_URL_PREAMBLE):
    print(f"Did not find build url preamble string '{BUILD_URL_PREAMBLE}' in"
          f" pipeline configuration from Buildkite API. Aborting.")
    sys.exit(3)

  previous_build_url = first_line[len(BUILD_URL_PREAMBLE):].strip()

  parsed_url = urllib.parse.urlparse(previous_build_url)
  path_components = parsed_url.path.split("/")
  # We're just going to be super picky here. If these invariants end up being a
  # problem, it's easy to relax them.
  if any((
      parsed_url.scheme != "https",
      parsed_url.netloc != "buildkite.com",
      parsed_url.params != "",
      parsed_url.query != "",
      parsed_url.fragment != "",
      len(path_components) != 5,
      # Path starts with a slash, so the first component is empty
      path_components[0] != "",
      path_components[3] != "builds",
  )):
    print(f"URL of build that previously updated the pipeline is not in"
          f" expected format. Got URL '{previous_build_url}'. Aborting")
    sys.exit(4)
  previous_organization = path_components[1]
  previous_pipeline = path_components[2]
  previous_build_number = int(path_components[4])

  if previous_organization != organization:
    print(f"Build was previously updated by a pipeline from a different"
          f"organization '{previous_organization}' not current organization"
          f" '{organization}'")
    sys.exit(5)
  if previous_pipeline != running_pipeline:
    print(f"Build was previously updated by a pipeline from a different"
          f"organization '{previous_pipeline}' not current organization"
          f" '{running_pipeline}'")
    sys.exit(5)

  if previous_build_number > running_build_number:
    print(f"...pipeline was already updated by later build"
          f" ({previous_build_number}) of this pipeline. Skipping update.")
    return False

  return True


def update_pipeline(bk, *, organization, pipeline_file, running_pipeline,
                    running_build_number, running_commit):
  pipeline_to_update, _ = os.path.splitext(os.path.basename(pipeline_file))

  short_running_commit = running_commit[:10]
  with open(pipeline_file) as f:
    new_pipeline_configuration = f.read()

  new_build_url = f"https://buildkite.com/{organization}/{running_pipeline}/builds/{running_build_number}"
  script_relpath = os.path.relpath(__file__)
  new_script_url = f"https://github.com/google/iree/blob/{short_running_commit}/{script_relpath}"
  new_pipeline_file_url = f"https://github.com/google/iree/blob/{short_running_commit}/{pipeline_file}"

  header = UPDATE_INFO_HEADER.format(build_url=new_build_url,
                                     script_url=new_script_url,
                                     pipeline_file_url=new_pipeline_file_url)
  new_pipeline_configuration = header + new_pipeline_configuration

  bk.pipelines().update_pipeline(organization=organization,
                                 pipeline=pipeline_to_update,
                                 configuration=new_pipeline_configuration)
  print("...updated successfully")


def parse_args():
  parser = argparse.ArgumentParser(
      description="Updates the configurations for all Buildkite pipeline.")
  parser.add_argument(
      "--force",
      action="store_true",
      default=False,
      help=("Force updates for all pipelines without checking the existing"
            " configuration. Use with caution."))
  parser.add_argument(
      "pipelines",
      type=str,
      nargs="*",
      help="Pipelines to update. Default is all of them.",
  )
  return parser.parse_args()


def main(args):
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
  running_pipeline = os.environ["BUILDKITE_PIPELINE_SLUG"]
  running_build_number = int(os.environ["BUILDKITE_BUILD_NUMBER"])
  running_commit = os.environ["BUILDKITE_COMMIT"]

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  git_root = get_git_root()
  os.chdir(git_root)
  glob_pattern = os.path.join(PIPELINE_ROOT_PATH, "*.yml")

  pipeline_files = ((
      os.path.join(PIPELINE_ROOT_PATH, f"{p}.yml") for p in args.pipelines)
                    if args.pipelines else glob.iglob(glob_pattern))
  first_error = None
  if args.force:
    print("Was passed force, so not checking existing pipeline configurations.")
  for pipeline_file in pipeline_files:
    # TODO: Support creating a new pipeline.
    print(f"Updating from: '{pipeline_file}'...")
    try:
      if args.force or should_update(bk,
                                     organization=organization,
                                     pipeline_file=pipeline_file,
                                     running_pipeline=running_pipeline,
                                     running_build_number=running_build_number):
        update_pipeline(
            bk,
            organization=organization,
            pipeline_file=pipeline_file,
            running_pipeline=running_pipeline,
            running_build_number=running_build_number,
            running_commit=running_commit,
        )
    except Exception as e:
      if first_error is None:
        first_error = e
      print(e)

  if first_error is not None:
    print("Encountered errors. Stack of first error:")
    raise first_error


if __name__ == "__main__":
  main(parse_args())
