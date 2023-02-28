#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import sys
import urllib.parse

from google.cloud import compute

CANARY_COMMAND_NAME = "canary"
ROLLBACK_CANARY_COMMAND_NAME = "rollback-canary"
PROMOTE_CANARY_COMMAND_NAME = "promote-canary"
DIRECT_UPDATE_COMMAND_NAME = "direct-update"

CANARY_SIZE = compute.FixedOrPercent(fixed=1)

TESTING_ENV_NAME = "testing"
PROD_ENV_NAME = "prod"


def resource_basename(resource):
  return os.path.basename(urllib.parse.urlparse(resource).path)


def error(msg):
  print("ERROR: ", msg, file=sys.stderr)
  sys.exit(1)


def confirm(msg):
  user_input = ""
  while user_input.lower() not in ["yes", "no", "y", "n"]:
    user_input = input(f"{msg} [y/n] ")
  if user_input.lower() in ["n", "no"]:
    print("Aborting")
    sys.exit(1)


def check_scary_action(action, skip_confirmation):
  if skip_confirmation:
    print(f"WARNING: Performing {action}.\n"
          f"Proceeding because '--skip-confirmation' is set.")
  else:
    confirm(f"You are about to perform {action}.\n"
            f" Are you sure you want to proceed?")


def summarize_versions(versions):
  return {v.name: resource_basename(v.instance_template) for v in versions}


class MigFetcher():

  def __init__(self, *, migs_client, regions_client, project):
    self._migs_client = migs_client
    self._regions_client = regions_client
    self._project = project

  def get_migs(self, *, region, type, group, prefix, modifier=None):
    print("Finding matching MIGs")
    migs = []

    request = compute.ListRegionsRequest(project=self._project)
    if region != "all":
      request.filter = f"name eq {region}"
    regions = [r.name for r in self._regions_client.list(request)]

    if type == "all":
      type = r"\w+"

    if group == "all":
      group = r"\w+"

    for region in regions:
      filter_parts = [p for p in [prefix, modifier, group, type, region] if p]
      filter = f"name eq '{'-'.join(filter_parts)}'"
      list_mig_request = compute.ListRegionInstanceGroupManagersRequest(
          project=self._project,
          region=region,
          filter=filter,
      )
      region_migs = self._migs_client.list(list_mig_request)
      migs.extend([mig for mig in region_migs])
    return migs


def main(args):
  templates_client = compute.InstanceTemplatesClient()
  migs_client = compute.RegionInstanceGroupManagersClient()
  updater = MigFetcher(
      migs_client=migs_client,
      regions_client=compute.RegionsClient(),
      project=args.project,
  )

  # Prod instances just have the bare name
  modifier = None if args.env == PROD_ENV_NAME else args.env
  migs = updater.get_migs(region=args.region,
                          type=args.type,
                          group=args.group,
                          prefix=args.name_prefix,
                          modifier=modifier)
  if len(migs) == 0:
    error("arguments matched no instance groups")
    sys.exit(1)

  print(f"Found:\n  ", "\n  ".join([m.name for m in migs]), sep="")
  if args.skip_confirmation:
    print("Proceeding with update as --skip-confirmation is set")
  else:
    confirm("Proceed with updating these MIGs?")

  if args.mode == "proactive" and args.action != "refresh":
    mig_desc = f"'{migs[0].name}'" if len(migs) == 1 else f"{len(migs)} groups"
    scary_action = (
        f"an update on {mig_desc} that will shut down instances even if"
        f" they're in the middle of running a job")
    check_scary_action(scary_action, args.skip_confirmation)

  for mig in migs:
    region = resource_basename(mig.region)
    if args.command in [DIRECT_UPDATE_COMMAND_NAME, CANARY_COMMAND_NAME]:
      if "testing" in args.version and args.env != TESTING_ENV_NAME:
        scary_action = (f"using testing template version '{args.version}' in"
                        f" environment '{args.env}'")
        check_scary_action(scary_action, args.skip_confirmation)

      strip = f"-{region}"
      if not mig.name.endswith(strip):
        raise ValueError(f"MIG name does not end with '{strip}' as expected")
      template_name = f"{mig.name[:-len(strip)]}-{args.version}"

      # TODO(gcmn): Make template naming consistent (ran into length limits)
      template_name = template_name.replace(f"-{args.env}-", "-")
      template_url = templates_client.get(
          project=args.project, instance_template=template_name).self_link

    current_templates = {v.name: v.instance_template for v in mig.versions}

    if not current_templates:
      error(f"Found no template versions for '{mig.name}'."
            f" This shouldn't be possible.")

    # TODO(gcmn): These should probably be factored into functions
    if args.command == CANARY_COMMAND_NAME:
      if len(current_templates) > 1:
        error(f"Instance group '{mig.name}' has multiple versions, but canary"
              f" requires it start with exactly one. Current versions:"
              f" {summarize_versions(mig.versions)}")

      base_template = current_templates.get(args.base_version_name)
      if not base_template:
        error(f"Instance group '{mig.name}' does not have a current version"
              f" named '{args.base_version_name}', which is required for an"
              f" automatic canary. Current versions:"
              f" {summarize_versions(mig.versions)}")

      if base_template == template_url:
        error(f"Instance group '{mig.name}' already has the requested canary"
              f" version '{template_name}' as its base version. Current"
              " versions:"
              f" {summarize_versions(mig.versions)}")
      new_versions = [
          compute.InstanceGroupManagerVersion(name=args.base_version_name,
                                              instance_template=base_template),
          compute.InstanceGroupManagerVersion(name=args.canary_version_name,
                                              instance_template=template_url,
                                              target_size=CANARY_SIZE)
      ]
    elif args.command == DIRECT_UPDATE_COMMAND_NAME:
      scary_action = (f"an update of all instances in '{mig.name}' directly"
                      f" without doing a canary")
      check_scary_action(scary_action, args.skip_confirmation)

      new_versions = [
          compute.InstanceGroupManagerVersion(name=args.base_version_name,
                                              instance_template=template_url)
      ]
    elif args.command == PROMOTE_CANARY_COMMAND_NAME:
      new_base_template = current_templates.get(args.canary_version_name)
      if new_base_template is None:
        error(f"Instance group '{mig.name}' does not have a current version"
              f" named '{args.canary_version_name}', which is required for an"
              f" automatic canary promotion. Current versions:"
              f" {summarize_versions(mig.versions)}")
      new_versions = [
          compute.InstanceGroupManagerVersion(
              name=args.base_version_name, instance_template=new_base_template)
      ]
    elif args.command == ROLLBACK_CANARY_COMMAND_NAME:
      base_template = current_templates.get(args.base_version_name)
      if base_template is None:
        error(f"Instance group '{mig.name}' does not have a current version"
              f" named '{args.base_version_name}', which is required for an"
              f" automatic canary rollback. Current versions:"
              f" {summarize_versions(mig.versions)}")
      new_versions = [
          compute.InstanceGroupManagerVersion(name=args.base_version_name,
                                              instance_template=base_template)
      ]
    else:
      error(f"Unrecognized command '{args.command}'")

    update_policy = compute.InstanceGroupManagerUpdatePolicy(
        type_=args.mode,
        minimal_action=args.action,
        most_disruptive_allowed_action=args.action)

    print(f"Updating {mig.name} to new versions:"
          f" {summarize_versions(new_versions)}")

    request = compute.PatchRegionInstanceGroupManagerRequest(
        project=args.project,
        region=region,
        instance_group_manager=mig.name,
        instance_group_manager_resource=compute.InstanceGroupManager(
            versions=new_versions, update_policy=update_policy))

    if not args.dry_run:
      migs_client.patch(request)
    else:
      print(f"Dry run, so not sending this patch request:\n```\n{request}```")
    print(f"Successfully updated {mig.name}")


def parse_args():
  parser = argparse.ArgumentParser(description=(
      "Updates one or more GCP Managed Instance Groups (MIGs) to new"
      " instance template versions. Wraps the GCP API with shortcuts for the"
      " patterns we have in our MIGs. See the README and"
      " https://cloud.google.com/compute/docs/instance-groups/updating-migs for"
      " more details."))

  # Makes global options come *after* command.
  # See https://stackoverflow.com/q/23296695
  subparser_base = argparse.ArgumentParser(add_help=False)
  subparser_base.add_argument("--project",
                              default="iree-oss",
                              help="The cloud project for the MIGs.")
  subparser_base.add_argument(
      "--region",
      "--regions",
      required=True,
      help=("The cloud region (e.g. 'us-west1') of the MIG to update, an RE2"
            " regex for matching region names (e.g. 'us-.*'), or 'all' to"
            " search for MIGs in all regions."))
  subparser_base.add_argument(
      "--group",
      "--groups",
      required=True,
      help=("The runner group of the MIGs to update, an RE2 regex for matching"
            " the group (e.g. 'cpu|gpu'), or 'all' to search for MIGs for all"
            " groups."),
  )
  subparser_base.add_argument(
      "--type",
      "--types",
      required=True,
      help=("The runner type of the MIGs to update, an RE2 regex for matching"
            " the type (e.g. 'presubmit|postsubmit'), or 'all' to search for"
            " MIGs for all types."),
  )
  subparser_base.add_argument(
      "--mode",
      default="opportunistic",
      choices=["opportunistic", "proactive"],
      help=(
          "The mode in which to update instances. See README and"
          " https://cloud.google.com/compute/docs/instance-groups/updating-migs."
      ))
  subparser_base.add_argument(
      "--action",
      choices=["refresh", "restart", "replace"],
      help=(
          "What action to take when updating an instance. See README and"
          " https://cloud.google.com/compute/docs/instance-groups/updating-migs."
      ))
  subparser_base.add_argument("--env",
                              "--environment",
                              default=TESTING_ENV_NAME,
                              help="The environment for the MIGs.",
                              choices=[PROD_ENV_NAME, TESTING_ENV_NAME])
  subparser_base.add_argument(
      "--dry-run",
      action="store_true",
      default=False,
      help="Print all output but don't actually send the update request.")

  # Defaulting to true for testing environment avoids people getting in the
  # habit of routinely passing --force.
  skip_confirmation = subparser_base.add_mutually_exclusive_group()
  skip_confirmation.add_argument(
      "--skip-confirmation",
      "--force",
      action="store_true",
      default=None,
      help=("Skip all confirmation prompts. Be careful."
            " Defaults to True for testing environment"))
  skip_confirmation.add_argument("--noskip-confirmation",
                                 "--noforce",
                                 action="store_false",
                                 default=None,
                                 dest="skip_confirmation")

  # These shouldn't be set very often, but it's just as easy to make them flags
  # as it is to make them global constants.
  subparser_base.add_argument("--name-prefix",
                              default="gh-runner",
                              help="The first part of MIG and template names.")
  subparser_base.add_argument(
      "--base-version-name",
      default="base",
      help="The name given to the MIG instance version that isn't in canary.")
  subparser_base.add_argument(
      "--canary-version-name",
      default="canary",
      help="The name given to the MIG instance version that is being canaried.")

  subparsers = parser.add_subparsers(required=True, dest="command")

  canary_sp = subparsers.add_parser(CANARY_COMMAND_NAME,
                                    parents=[subparser_base],
                                    help="Canary a new template version.")
  rollback_sp = subparsers.add_parser(
      ROLLBACK_CANARY_COMMAND_NAME,
      parents=[subparser_base],
      help=("Rollback a previous canary, restoring all instances to the base"
            " version."))
  promote_sp = subparsers.add_parser(
      PROMOTE_CANARY_COMMAND_NAME,
      parents=[subparser_base],
      help="Promote the current canary version to be the base version.")
  direct_sp = subparsers.add_parser(
      DIRECT_UPDATE_COMMAND_NAME,
      parents=[subparser_base],
      help=("Update all instances in the MIG to a new version. Generally should"
            " not be used for prod."))

  for sp in [canary_sp, direct_sp]:
    sp.add_argument(
        "--version",
        help=("The new instance template version. Usually git hash +"
              " 3-character uid, e.g. 56e40f6505-9lp"))

  # TODO: Add this argument with a custom parser
  # canary_sp.add_argument("--canary-size", type=int, default=1)

  args = parser.parse_args()

  if args.skip_confirmation is None:
    args.skip_confirmation = args.env == TESTING_ENV_NAME

  if args.action is None:
    if args.mode == "proactive":
      args.action = "refresh"
    else:
      args.action = "replace"

  return args


if __name__ == "__main__":
  main(parse_args())
