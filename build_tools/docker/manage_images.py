#!/usr/bin/env python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Manages IREE Docker image definitions.

Includes information on their dependency graph and GCR URL.

See the README for more information on how to add and update images.

Example usage:

Rebuild the cmake image and all images that transitively on depend on it,
tagging them with `latest` and updating all references to their sha digests:
  python3 build_tools/docker/manage_images.py --image cmake

Print out output for rebuilding the cmake image and all images that
transitively depend on it, but don't take side-effecting actions:
  python3 build_tools/docker/manage_images.py --image cmake --dry-run

Rebuild and push all images and update references to them in the repository:
  python3 build_tools/docker/manage_images.py --images all
"""

import argparse
import fileinput
import os
import posixpath
import re
import subprocess
import sys
from typing import Dict, List, Sequence, Union

import utils


IREE_GCR_URL = "gcr.io/iree-oss/"
DIGEST_REGEX = r"sha256:[a-zA-Z0-9]+"
DOCKER_DIR = "build_tools/docker/".replace("/", os.sep)

# Map from image names to images that they depend on.
IMAGES_TO_DEPENDENCIES = {
    "base": [],
    "base-arm64": [],
    "android": ["base"],
    "emscripten": ["base"],
    "nvidia": ["base"],
    "riscv": ["base"],
    "riscv-toolchain-builder": [],
    "gradle-android": ["base"],
    "frontends": ["android"],
    "shark": [],
    "swiftshader": ["base"],
    "samples": ["swiftshader"],
    "frontends-swiftshader": ["frontends", "swiftshader"],
    "frontends-nvidia": ["frontends"],
    # Containers with all the newest versions of dependencies that we support
    # instead of the oldest.
    "base-bleeding-edge": [],
    "swiftshader-bleeding-edge": ["base-bleeding-edge"],
    "nvidia-bleeding-edge": ["base-bleeding-edge"],
}

IMAGES_TO_DEPENDENT_IMAGES = {k: [] for k in IMAGES_TO_DEPENDENCIES}
for image, dependencies in IMAGES_TO_DEPENDENCIES.items():
    for dependency in dependencies:
        IMAGES_TO_DEPENDENT_IMAGES[dependency].append(image)

IMAGES_HELP = [f"`{name}`" for name in IMAGES_TO_DEPENDENCIES]
IMAGES_HELP = f"{', '.join(IMAGES_HELP)} or `all`"


def parse_arguments():
    """Parses command-line options."""
    parser = argparse.ArgumentParser(
        description="Build IREE's Docker images and optionally push them to GCR."
    )
    parser.add_argument(
        "--images",
        "--image",
        type=str,
        required=True,
        action="append",
        help=f"Name of the image to build: {IMAGES_HELP}.",
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        "-n",
        action="store_true",
        help="Print output without building or pushing any images.",
    )
    parser.add_argument(
        "--only_references",
        "--only-references",
        action="store_true",
        help="Just update references to images using the digests in prod_digests.txt",
    )

    args = parser.parse_args()
    for image in args.images:
        if image == "all":
            # Sort for a determinstic order
            args.images = sorted(IMAGES_TO_DEPENDENCIES.keys())
        elif image not in IMAGES_TO_DEPENDENCIES:
            raise parser.error(
                "Expected --image to be one of:\n"
                f"  {IMAGES_HELP}\n"
                f"but got `{image}`."
            )
    return args


def _dag_dfs(
    input_nodes: Sequence[str], node_to_child_nodes: Dict[str, Sequence[str]]
) -> List[str]:
    # Python doesn't have a builtin OrderedSet, but we don't have many images, so
    # we just use a list.
    ordered_nodes = []

    def add_children(parent_node: str):
        if parent_node not in ordered_nodes:
            for child_node in node_to_child_nodes[parent_node]:
                add_children(child_node)
            ordered_nodes.append(parent_node)

    for node in input_nodes:
        add_children(node)
    return ordered_nodes


def get_ordered_images_to_process(images: Sequence[str]) -> List[str]:
    dependents = _dag_dfs(images, IMAGES_TO_DEPENDENT_IMAGES)
    dependents.reverse()
    return dependents


def get_dependencies(images: Sequence[str]) -> List[str]:
    return _dag_dfs(images, IMAGES_TO_DEPENDENCIES)


def get_repo_digest(tagged_image_url: str, dry_run: bool = False) -> str:
    inspect_command = [
        "docker",
        "image",
        "inspect",
        tagged_image_url,
        "-f",
        "{{index .RepoDigests 0}}",
    ]
    try:
        completed_process = utils.run_command(
            inspect_command,
            dry_run=False,  # Run even if --dry_run is True.
            capture_output=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as error:
        if dry_run:
            return ""
        else:
            raise RuntimeError(
                f"Computing the repository digest for {tagged_image_url} failed. Has "
                "it been pushed to GCR?"
            ) from error
    _, repo_digest = completed_process.stdout.strip().split("@")
    return repo_digest


def update_references(image_url: str, digest: str, dry_run: bool = False):
    """Updates all references to "image_url" with a sha256 digest."""
    print(f"Updating references to {image_url}")

    grep_command = ["git", "grep", "-l", f"{image_url}@sha256"]
    try:
        completed_process = utils.run_command(
            grep_command, capture_output=True, timeout=5
        )
    except subprocess.CalledProcessError as error:
        if error.returncode == 1:
            print(f"Found no references to {image_url}")
            return
        raise error

    # Update references in all grepped files.
    files = completed_process.stdout.split()
    print(f"Updating references in {len(files)} files: {files}")
    if not dry_run:
        for line in fileinput.input(files=files, inplace=True):
            print(
                re.sub(f"{image_url}@{DIGEST_REGEX}", f"{image_url}@{digest}", line),
                end="",
            )


def parse_prod_digests() -> Dict[str, str]:
    image_urls_to_prod_digests = {}
    with open(utils.PROD_DIGESTS_PATH, "r") as f:
        for line in f:
            image_url, digest = line.strip().split("@")
            image_urls_to_prod_digests[image_url] = digest
    return image_urls_to_prod_digests


if __name__ == "__main__":
    args = parse_arguments()
    image_urls_to_prod_digests = parse_prod_digests()
    images_to_process = get_ordered_images_to_process(args.images)
    print(f"Also processing dependent images. Will process: {images_to_process}")

    if not args.only_references:
        # Ensure the user has the correct authorization to push to GCR.
        utils.check_gcloud_auth(dry_run=args.dry_run)

        dependencies = get_dependencies(images_to_process)
        print(f"Pulling image dependencies: {dependencies}")
        for dependency in dependencies:
            dependency_url = posixpath.join(IREE_GCR_URL, dependency)
            # If `dependency` is a new image then it may not have a prod digest yet.
            if dependency_url in image_urls_to_prod_digests:
                digest = image_urls_to_prod_digests[dependency_url]
                dependency_with_digest = f"{dependency_url}@{digest}"
                utils.run_command(
                    ["docker", "pull", dependency_with_digest], dry_run=args.dry_run
                )

    for image in images_to_process:
        print("\n" * 5 + f"Processing image {image}")
        image_url = posixpath.join(IREE_GCR_URL, image)
        tagged_image_url = f"{image_url}"
        image_path = os.path.join(DOCKER_DIR, "dockerfiles", f"{image}.Dockerfile")

        if args.only_references:
            digest = image_urls_to_prod_digests[image_url]
        else:
            # We deliberately give the whole repository as context so we can reuse
            # scripts and such. It would be nice if Docker gave us a way to make this
            # more explicit, like symlinking files in the context, but they refuse to
            # with the justification that it makes builds non-hermetic, a hilarious
            # concern for something that allows and encourages arbitrary network
            # access in builds.
            # We're assuming this is being run from the root of the repository.
            # FIXME: make this more robust to where it is run from.
            utils.run_command(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--file",
                    image_path,
                    "--load",
                    "--tag",
                    tagged_image_url,
                    ".",
                ],
                dry_run=args.dry_run,
            )

            utils.run_command(
                ["docker", "push", tagged_image_url], dry_run=args.dry_run
            )

            digest = get_repo_digest(tagged_image_url, args.dry_run)

            # Check that the image is in "prod_digests.txt" and append it to the list
            # in the file if it isn't.
            if image_url not in image_urls_to_prod_digests:
                image_with_digest = f"{image_url}@{digest}"
                print(
                    f"Adding new image {image_with_digest} to {utils.PROD_DIGESTS_PATH}"
                )
                if not args.dry_run:
                    with open(utils.PROD_DIGESTS_PATH, "a") as f:
                        f.write(f"{image_with_digest}\n")

        update_references(image_url, digest, dry_run=args.dry_run)
