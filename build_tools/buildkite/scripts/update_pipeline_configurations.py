#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Updates all Buildkite pipelines with the Buildkite API.

This overwrites the configuration of all the Buildkite pipelines based on the
pipeline files under the `trusted/` and `untrusted/` directories. Pipelines
should be updated this way, not in the UI. The pipeline configuration uploaded
is one that bootstraps the appropriate configuration from the repository. Before
updating, this checks for a specific header in the configuration stored in the
API that indicates which Buildkite build previously updated it. If the updater
was not this same pipeline the script fails. If it was a later build in this
pipeline, it skips updating unless the `--force` flag is passed. This is to
avoid situations where an earlier build is for some reason running a step after
a later build due to race conditions or a retry. Buildkite concurrency groups
should be used to prevent builds from trying to update pipelines simultaneously.
"""

import argparse
import glob
import os
import subprocess
import sys
import urllib

import requests
from pybuildkite import buildkite

from common.buildkite_utils import get_pipeline

GIT_REPO = "https://github.com/google/iree"
PIPELINE_ROOT_PATH = "build_tools/buildkite/pipelines"
TRUSTED_BOOTSTRAP_PIPELINE_PATH = os.path.join(PIPELINE_ROOT_PATH, "fragment",
                                               "bootstrap-trusted.yml")
UNTRUSTED_BOOTSTRAP_PIPELINE_PATH = os.path.join(PIPELINE_ROOT_PATH, "fragment",
                                                 "bootstrap-untrusted.yml")

IREE_TEAM_REST_UUID = "1a2cbc72-2c8e-4375-821e-e3dfa1db96b5"
# The team that just contains the postsubmit bot
PRIVILEGED_TEAM_REST_UUID = "7de7685f-7e10-4bc4-ba66-3478a5c56db4"

FORCE_UPDATE_ENV_VAR = "IREE_FORCE_BUILDKITE_PIPELINE_UPDATE"

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


def prepend_header(configuration, *, organization, running_pipeline,
                   running_commit, running_build_number, trusted):
  short_running_commit = running_commit[:10]

  new_build_url = f"https://buildkite.com/{organization}/{running_pipeline}/builds/{running_build_number}"
  script_relpath = os.path.relpath(__file__)
  new_script_url = f"https://github.com/google/iree/blob/{short_running_commit}/{script_relpath}"

  bootstrap_pipeline_path = TRUSTED_BOOTSTRAP_PIPELINE_PATH if trusted else UNTRUSTED_BOOTSTRAP_PIPELINE_PATH

  new_pipeline_file_url = f"https://github.com/google/iree/blob/{short_running_commit}/{bootstrap_pipeline_path}"

  header = UPDATE_INFO_HEADER.format(build_url=new_build_url,
                                     script_url=new_script_url,
                                     pipeline_file_url=new_pipeline_file_url)
  return header + configuration


def should_update(bk, *, organization, configuration, existing_pipeline,
                  running_pipeline, running_build_number):
  previous_configuration_lines = existing_pipeline["configuration"].splitlines()
  trimmed_previous_configuration_lines = previous_configuration_lines[
      len(UPDATE_INFO_HEADER.splitlines()):]
  if trimmed_previous_configuration_lines == configuration.splitlines():
    print("Configuration has not changed. Not updating")
    return False

  first_line = previous_configuration_lines[0].strip()
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
          f" organization '{previous_organization}' not current organization"
          f" '{organization}'")
    sys.exit(5)
  if previous_pipeline != running_pipeline:
    print(f"Build was previously updated by a different pipeline"
          f" '{previous_pipeline}' not current pipeline '{running_pipeline}'")
    sys.exit(5)

  if previous_build_number > running_build_number:
    print(f"...pipeline was already updated by later build"
          f" ({previous_build_number}) of this pipeline. Skipping update.")
    return False

  return True


def create_pipeline(bk, *, organization, pipeline_slug, configuration,
                    running_pipeline, running_build_number, running_commit,
                    trusted, dry_run):
  configuration = prepend_header(configuration,
                                 organization=organization,
                                 running_pipeline=running_pipeline,
                                 running_build_number=running_build_number,
                                 running_commit=running_commit,
                                 trusted=trusted)

  # TODO: Update pybuildkite to allow passing provider_settings.
  # see https://github.com/pyasi/pybuildkite/issues/73
  pipelines_api = bk.pipelines()

  data = {
      "name": pipeline_slug,
      "repository": GIT_REPO,
      "configuration": configuration,
      # With the rest API, we can only give "Full Access", so this is limited to
      # the IREE team and doesn't give Read & Build access to the "Everyone" and
      # "Presubmit" teams. I'm talking to Buildkite support about this
      # limitation. There's a similar problem that the pipeline can't be made
      # public via the REST API, which requires user intervention in the UI.
      "team_uuids": [IREE_TEAM_REST_UUID, PRIVILEGED_TEAM_REST_UUID],
      "provider_settings": {
          # We don't want any automatic triggering from GitHub webhooks.
          "trigger_mode": "none"
      },
  }

  print(f"Creating pipeline {pipeline_slug} with payload:\n"
        f"```\n"
        f"{configuration}\n"
        f"```\n")
  if not dry_run:
    pipelines_api.client.post(pipelines_api.path.format("iree"), body=data)

  print("...created successfully")


def update_pipeline(bk, *, organization, pipeline_slug, configuration,
                    running_pipeline, running_build_number, running_commit,
                    trusted, dry_run):
  configuration = prepend_header(configuration,
                                 organization=organization,
                                 running_pipeline=running_pipeline,
                                 running_build_number=running_build_number,
                                 running_commit=running_commit,
                                 trusted=trusted)

  print(f"Updating pipeline {pipeline_slug} with configuration:\n"
        f"```\n"
        f"{configuration}\n"
        f"```\n")
  if not dry_run:
    bk.pipelines().update_pipeline(organization=organization,
                                   pipeline=pipeline_slug,
                                   configuration=configuration)
  print("...updated successfully")


def get_slug(pipeline_file):
  pipeline_slug, _ = os.path.splitext(os.path.basename(pipeline_file))
  return pipeline_slug


def update_pipelines(bk, pipeline_files, *, organization, running_pipeline,
                     running_build_number, running_commit, trusted, force,
                     dry_run):
  first_error = None
  for pipeline_file in pipeline_files:
    pipeline_slug = get_slug(pipeline_file)

    try:
      with open(TRUSTED_BOOTSTRAP_PIPELINE_PATH
                if trusted else UNTRUSTED_BOOTSTRAP_PIPELINE_PATH) as f:
        configuration = f.read()
      existing_pipeline = get_pipeline(bk,
                                       organization=organization,
                                       pipeline_slug=pipeline_slug)
      if existing_pipeline is None:
        print(f"Creating for: '{pipeline_file}'...")
        create_pipeline(bk,
                        organization=organization,
                        pipeline_slug=pipeline_slug,
                        configuration=configuration,
                        running_pipeline=running_pipeline,
                        running_build_number=running_build_number,
                        running_commit=running_commit,
                        trusted=trusted,
                        dry_run=dry_run)
        continue

      print(f"Updating for: '{pipeline_file}'...")
      if force or should_update(bk,
                                organization=organization,
                                configuration=configuration,
                                existing_pipeline=existing_pipeline,
                                running_pipeline=running_pipeline,
                                running_build_number=running_build_number):
        update_pipeline(
            bk,
            organization=organization,
            pipeline_slug=pipeline_slug,
            configuration=configuration,
            running_pipeline=running_pipeline,
            running_build_number=running_build_number,
            running_commit=running_commit,
            trusted=trusted,
            dry_run=dry_run,
        )
    except Exception as e:
      if first_error is None:
        first_error = e
      print(e)
  return first_error


def parse_args():
  force_update = os.environ.get(FORCE_UPDATE_ENV_VAR)
  force_update = not (force_update is None or force_update == "0" or
                      force_update.lower() == "false" or force_update == "")
  parser = argparse.ArgumentParser(
      description="Updates the configurations for all Buildkite pipelines.")
  parser.add_argument(
      "--force",
      action="store_true",
      default=force_update,
      help=(f"Force updates for all pipelines without checking the existing"
            f" configuration. Use with caution. Can also be set via the"
            f" environment variable {FORCE_UPDATE_ENV_VAR}."))
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Don't actually update or create any pipelines.")
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
  repository = os.environ["BUILDKITE_REPO"]
  running_pipeline = os.environ["BUILDKITE_PIPELINE_SLUG"]
  running_build_number = int(os.environ["BUILDKITE_BUILD_NUMBER"])
  running_commit = os.environ["BUILDKITE_COMMIT"]

  bk = buildkite.Buildkite()
  bk.set_access_token(access_token)

  git_root = get_git_root()
  os.chdir(git_root)

  trusted_pipeline_files = glob.iglob(
      os.path.join(PIPELINE_ROOT_PATH, "trusted", "*.yml"))
  untrusted_pipeline_files = glob.iglob(
      os.path.join(PIPELINE_ROOT_PATH, "untrusted", "*.yml"))

  if args.pipelines:
    trusted_pipeline_files = (
        p for p in trusted_pipeline_files if get_slug(p) in args.pipelines)
    untrusted_pipeline_files = (
        p for p in untrusted_pipeline_files if get_slug(p) in args.pipelines)

  if args.force:
    print("Was passed force, so not checking existing pipeline configurations.")

  first_error = update_pipelines(bk,
                                 trusted_pipeline_files,
                                 organization=organization,
                                 running_pipeline=running_pipeline,
                                 running_build_number=running_build_number,
                                 running_commit=running_commit,
                                 trusted=True,
                                 force=args.force,
                                 dry_run=args.dry_run)
  first_error = (first_error or
                 update_pipelines(bk,
                                  untrusted_pipeline_files,
                                  organization=organization,
                                  running_pipeline=running_pipeline,
                                  running_build_number=running_build_number,
                                  running_commit=running_commit,
                                  trusted=False,
                                  force=args.force,
                                  dry_run=args.dry_run))

  if first_error is not None:
    print("Encountered errors. Stack of first error:")
    raise first_error


if __name__ == "__main__":
  main(parse_args())
