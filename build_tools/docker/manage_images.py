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
import json
import pathlib
import re
import subprocess
from typing import Dict, List, Sequence

import utils

IREE_GCR_URL = "gcr.io/iree-oss/"
DIGEST_REGEX = r"sha256:[a-zA-Z0-9]+"
IMAGE_DEPS_FILENAME = "image_deps.json"
PROD_DIGESTS_FILENAME = "prod_digests.txt"
DOCKERFILES_DIRNAME = "dockerfiles"


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
        help="Name of the image to build",
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        "-n",
        action="store_true",
        help="Print output without building or pushing any images",
    )
    parser.add_argument(
        "--only_references",
        "--only-references",
        action="store_true",
        help="Just update references to images using the digests in --image_deps",
    )
    parser.add_argument(
        "--docker_dir",
        "--docker-dir",
        type=pathlib.Path,
        default=pathlib.Path("build_tools/docker"),
        help=(
            "Directory that contains: `dockerfiles/*.Dockerfile`,"
            " `image_deps.json`, and `prod_digests.txt`"
        ),
    )

    args = parser.parse_args()
    return args


def _dag_dfs(
    input_nodes: Sequence[str], node_to_child_nodes: Dict[str, List[str]]
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


def _get_images_to_dependents(
    images_to_dependencies: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    images_to_dependents = {k: [] for k in images_to_dependencies.keys()}
    for image, dependencies in images_to_dependencies.items():
        for dependency in dependencies:
            images_to_dependents[dependency].append(image)

    return images_to_dependents


def get_dependencies(
    images: Sequence[str], images_to_dependents: Dict[str, List[str]]
) -> List[str]:
    return _dag_dfs(input_nodes=images, node_to_child_nodes=images_to_dependents)


def get_ordered_images_to_process(
    images: Sequence[str],
    images_to_dependents: Dict[str, List[str]],
) -> List[str]:
    dependents = get_dependencies(
        images=images, images_to_dependents=images_to_dependents
    )
    dependents.reverse()
    return dependents


def parse_prod_digests(prod_digests_path: pathlib.Path) -> Dict[str, str]:
    image_urls_to_prod_digests = {}
    for line in prod_digests_path.read_text().splitlines():
        image_url, digest = line.strip().split("@")
        image_urls_to_prod_digests[image_url] = digest
    return image_urls_to_prod_digests


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
        raise RuntimeError(
            f"Computing the repository digest for {tagged_image_url} failed."
            " Has it been pushed to GCR?"
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


def main(
    images: Sequence[str],
    only_references: bool,
    docker_dir: pathlib.Path,
    dry_run: bool,
):
    image_deps = json.loads((docker_dir / IMAGE_DEPS_FILENAME).read_text())
    if "all" in images:
        images = list(image_deps.keys())
    for image in images:
        if image not in image_deps:
            raise ValueError(
                f'Image "{image}" not found in the image graph.'
                f' Available options: all,{",".join(image_deps.keys())}'
            )

    images_to_dependents = _get_images_to_dependents(image_deps)
    images_to_process = get_ordered_images_to_process(
        images=images, images_to_dependents=images_to_dependents
    )
    prod_digests_path = docker_dir / PROD_DIGESTS_FILENAME
    image_urls_to_prod_digests = parse_prod_digests(prod_digests_path)
    print(f"Also processing dependent images. Will process: {images_to_process}")

    if not only_references:
        # Ensure the user has the correct authorization to push to GCR.
        utils.check_gcloud_auth(dry_run=dry_run)

        dependencies = get_dependencies(
            images=images_to_process, images_to_dependents=images_to_dependents
        )
        print(f"Pulling image dependencies: {dependencies}")
        for dependency in dependencies:
            image_digest = image_urls_to_prod_digests.get(f"{IREE_GCR_URL}{dependency}")
            # If `dependency` is a new image then it may not have a digest yet.
            if image_digest is not None:
                dependency_url = f"{IREE_GCR_URL}{dependency}"
                dependency_with_digest = f"{dependency_url}@{image_digest}"
                utils.run_command(
                    ["docker", "pull", dependency_with_digest], dry_run=dry_run
                )

    for image in images_to_process:
        print("\n" * 5 + f"Processing image {image}")
        image_url = f"{IREE_GCR_URL}{image}"
        image_path = docker_dir / DOCKERFILES_DIRNAME / f"{image}.Dockerfile"

        if only_references:
            image_digest = image_urls_to_prod_digests.get(image_url)
            if image_digest is None:
                raise ValueError(
                    f"Can't update the references of {image} because it has no digest."
                )
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
                    "build",
                    "--file",
                    image_path,
                    "--tag",
                    image_url,
                    ".",
                ],
                dry_run=dry_run,
            )

            utils.run_command(["docker", "push", image_url], dry_run=dry_run)

            image_digest = get_repo_digest(image_url, dry_run)

            # Check that the image is in "prod_digests.txt" and append it to the
            # list in the file if it isn't.
            if image_url not in image_urls_to_prod_digests:
                image_with_digest = f"{image_url}@{image_digest}"
                print(f"Adding new image {image_with_digest} to {prod_digests_path}")
                if not dry_run:
                    with prod_digests_path.open("a") as digests_file:
                        digests_file.write(f"{image_with_digest}\n")

        update_references(image_url, image_digest, dry_run=dry_run)


if __name__ == "__main__":
    main(**vars(parse_arguments()))
