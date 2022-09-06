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

GROUPS = ["presubmit", "postsubmit"]
TYPES = ["cpu", "gpu"]


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


def summarize_versions(versions):
  return {v.name: resource_basename(v.instance_template) for v in versions}


class MigFetcher():

  def __init__(self, *, migs_client, regions_client, project):
    self._migs_client = migs_client
    self._regions_client = regions_client
    self._project = project

  def get_migs(self, *, regions, types, groups, prefix, modifier=None):
    print("Finding matching MIGs")
    migs = []
    if isinstance(regions, str):
      if regions == "all":
        # Yeah apparently we have to iterate through regions ourselves
        regions = [
            r.name for r in self._regions_client.list(project=self._project)
        ]
      else:
        regions = [regions]

    if isinstance(types, str):
      if types == "all":
        types = [".*"]
      else:
        types = [types]
    types_filter = f"({'|'.join(types)})"

    groups_filter = groups
    if isinstance(groups, str):
      if groups == "all":
        groups = [".*"]
      else:
        groups = [groups]
    groups_filter = f"({'|'.join(groups)})"

    for region in regions:
      # type_filter = ".*"
      filter_parts = [
          p for p in [prefix, modifier, groups_filter, types_filter, region]
          if p
      ]
      filter = f"name eq {'-'.join(filter_parts)}"
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

  modifier = args.testing_name_modifier if args.testing else None
  migs = updater.get_migs(regions=args.region,
                          types=args.type,
                          groups=args.group,
                          prefix=args.name_prefix,
                          modifier=modifier)
  if len(migs) == 0:
    error("arguments matched no instance groups")
    sys.exit(1)

  print(f"Found:\n  ", "\n  ".join([m.name for m in migs]), sep="")
  confirm("Proceed with updating these MIGs?")

  if args.mode == "proactive" and args.action != "refresh":
    warning_text = f"'{migs[0].name}'" if len(
        migs) == 1 else f"{len(migs)} groups"
    confirm(f"You are about to perform an update on {warning_text} that will"
            f" shut down instances even if they're in the middle of running a"
            f" job. Are you sure you want to proceed?")

  for mig in migs:
    region = resource_basename(mig.region)
    if args.command in [DIRECT_UPDATE_COMMAND_NAME, CANARY_COMMAND_NAME]:
      strip = f"-{region}"
      if not mig.name.endswith(strip):
        raise ValueError(f"MIG name does not end with '{strip}' as expected")
      template_name = f"{mig.name[:-len(strip)]}-{args.version}"

      # TODO: Make testing template naming consistent (ran into length limits)
      if args.testing:
        template_name = template_name.replace(f"-{args.testing_name_modifier}-",
                                              "-")
      template_url = templates_client.get(
          project=args.project, instance_template=template_name).self_link

    current_templates = {v.name: v.instance_template for v in mig.versions}

    if not current_templates:
      error(f"Found no template versions for '{mig.name}'."
            f" This shouldn't be possible.")

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

      new_versions = [
          compute.InstanceGroupManagerVersion(name=args.base_version_name,
                                              instance_template=base_template),
          compute.InstanceGroupManagerVersion(name=args.canary_version_name,
                                              instance_template=template_url,
                                              target_size=CANARY_SIZE)
      ]
    elif args.command == DIRECT_UPDATE_COMMAND_NAME:
      confirm(f"You are about to update all instances in '{mig.name}' directly"
              f" without doing a canary. Are you sure you want to proceed?")
      new_versions = [
          compute.InstanceGroupManagerVersion(name=args.base_version_name,
                                              instance_template=template_url)
      ]
    elif args.command == PROMOTE_CANARY_COMMAND_NAME:
      new_base_template = current_templates.get(args.canary_version_name)
      new_versions = [
          compute.InstanceGroupManagerVersion(
              name=args.base_version_name, instance_template=new_base_template)
      ]
    elif args.command == ROLLBACK_CANARY_COMMAND_NAME:
      base_template = current_templates.get(args.base_version_name)
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

    migs_client.patch(
        project=args.project,
        region=region,
        instance_group_manager=mig.name,
        instance_group_manager_resource=compute.InstanceGroupManager(
            versions=new_versions, update_policy=update_policy))
    print(f"Successfully updated {mig.name}")


def parse_args():
  parser = argparse.ArgumentParser()

  # Makes global options come *after* command.
  # See https://stackoverflow.com/q/23296695
  subparser_base = argparse.ArgumentParser(add_help=False)
  subparser_base.add_argument("--project", default="iree-oss")
  subparser_base.add_argument("--region")
  subparser_base.add_argument("--group", required=True)
  subparser_base.add_argument("--type", required=True)
  subparser_base.add_argument("--mode", choices=["opportunistic", "proactive"])
  subparser_base.add_argument("--action",
                              default="refresh",
                              choices=["refresh", "restart", "replace"])
  # These shouldn't be common, but it's just as easy to make them flags as it is
  # to make them global constants.
  subparser_base.add_argument("--testing-name-modifier", default="testing")
  subparser_base.add_argument("--name-prefix", default="github-runner")
  subparser_base.add_argument("--base-version-name", default="base")
  subparser_base.add_argument("--canary-version-name", default="canary")

  testing = subparser_base.add_mutually_exclusive_group()
  testing.add_argument("--testing", action="store_true", default=True)
  testing.add_argument("--prod", dest="testing", action="store_false")

  subparsers = parser.add_subparsers(required=True, dest="command")

  canary_sp = subparsers.add_parser(CANARY_COMMAND_NAME,
                                    parents=[subparser_base])
  rollback_sp = subparsers.add_parser(ROLLBACK_CANARY_COMMAND_NAME,
                                      parents=[subparser_base])
  promote_sp = subparsers.add_parser(PROMOTE_CANARY_COMMAND_NAME,
                                     parents=[subparser_base])
  direct_sp = subparsers.add_parser(DIRECT_UPDATE_COMMAND_NAME,
                                    parents=[subparser_base])

  for sp in [canary_sp, direct_sp]:
    sp.add_argument("--version")

  # TODO: Add this argument with a custom parser
  # canary_sp.add_argument("--canary-size", type=int, default=1)

  args = parser.parse_args()

  if args.mode is None:
    if args.action == "refresh":
      args.mode = "proactive"
    else:
      args.mode = "opportunistic"

  return args


if __name__ == "__main__":
  main(parse_args())
