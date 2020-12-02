#!/usr/bin/env python3

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Manages IREE Docker image definitions.

Includes information on their dependency graph and GCR URL.

See the README for more information on how to add and update images.

Example usage:

Rebuild the cmake image and all images that transitively on depend on it,
tagging them with `latest` and updating all references to their sha digests:
  python3 build_tools/docker/manage_images.py --image cmake

Print out output for rebuilding the cmake image and all images that
transitively on depend on it, but don't take side-effecting actions:
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

IREE_GCR_URL = 'gcr.io/iree-oss/'
DIGEST_REGEX = r'sha256:[a-zA-Z0-9]+'
DOCKER_DIR = 'build_tools/docker/'.replace('/', os.sep)

# Map from image names to images that they depend on.
IMAGES_TO_DEPENDENCIES = {
    'base': [],
    'bazel': ['base', 'util'],
    'bazel-python': ['bazel'],
    'bazel-tensorflow': ['bazel-python'],
    'bazel-tensorflow-nvidia': ['bazel-tensorflow-vulkan'],
    'bazel-tensorflow-swiftshader': ['bazel-tensorflow-vulkan', 'swiftshader'],
    'bazel-tensorflow-vulkan': ['bazel-tensorflow', 'vulkan'],
    'cmake': ['base', 'util'],
    'cmake-android': ['cmake', 'util'],
    'cmake-python': ['cmake'],
    'cmake-python-nvidia': ['cmake-python-vulkan'],
    'cmake-python-swiftshader': ['cmake-python-vulkan', 'swiftshader'],
    'cmake-python-vulkan': ['cmake-python', 'vulkan'],
    'rbe-toolchain': ['vulkan'],
    'swiftshader': ['cmake'],
    'util': [],
    'vulkan': ['util'],
}

IMAGES_TO_DEPENDENT_IMAGES = {k: [] for k in IMAGES_TO_DEPENDENCIES}
for image, dependencies in IMAGES_TO_DEPENDENCIES.items():
  for dependency in dependencies:
    IMAGES_TO_DEPENDENT_IMAGES[dependency].append(image)

IMAGES_HELP = [f'`{name}`' for name in IMAGES_TO_DEPENDENCIES]
IMAGES_HELP = f'{", ".join(IMAGES_HELP)} or `all`'


def parse_arguments():
  """Parses command-line options."""
  parser = argparse.ArgumentParser(
      description="Build IREE's Docker images and optionally push them to GCR.")
  parser.add_argument('--images',
                      '--image',
                      type=str,
                      required=True,
                      action='append',
                      help=f'Name of the image to build: {IMAGES_HELP}.')
  parser.add_argument(
      '--dry_run',
      '--dry-run',
      '-n',
      action='store_true',
      help='Print output without building or pushing any images.')

  args = parser.parse_args()
  for image in args.images:
    if image == 'all':
      # Sort for a determinstic order
      args.images = sorted(IMAGES_TO_DEPENDENCIES.keys())
    elif image not in IMAGES_TO_DEPENDENCIES:
      raise parser.error('Expected --image to be one of:\n'
                         f'  {IMAGES_HELP}\n'
                         f'but got `{image}`.')
  return args


def get_ordered_images_to_process(images: Sequence[str]) -> List[str]:
  # Python doesn't have a builtin OrderedSet, so we mimic one to the extent
  # that we need by using 'in' before adding any elements.
  processing_order = []

  def add_dependent_images(image: str):
    if image not in processing_order:
      for dependent_image in IMAGES_TO_DEPENDENT_IMAGES[image]:
        add_dependent_images(dependent_image)
      processing_order.append(image)

  for image in images:
    add_dependent_images(image)

  processing_order.reverse()
  return processing_order


def get_dependencies(images: Sequence[str]) -> Sequence[str]:
  dependencies = set()

  def add_dependency(image: str):
    if image not in dependencies:
      for dependency in IMAGES_TO_DEPENDENCIES[image]:
        add_dependency(dependency)
      dependencies.add(image)

  for image in images:
    add_dependency(image)

  return dependencies


def run_command(command: Sequence[str],
                dry_run: bool = False,
                check: bool = True,
                capture_output: bool = False,
                universal_newlines: bool = True,
                **run_kwargs) -> subprocess.CompletedProcess:
  """Thin wrapper around subprocess.run"""
  print(f'Running: `{" ".join(command)}`')
  if dry_run:
    # Dummy CompletedProess with successful returncode.
    return subprocess.CompletedProcess(command, returncode=0)

  if capture_output:
    # Hardcode support for python <= 3.6.
    run_kwargs['stdout'] = subprocess.PIPE
    run_kwargs['stderr'] = subprocess.PIPE
  return subprocess.run(command,
                        universal_newlines=universal_newlines,
                        check=check,
                        **run_kwargs)


def get_repo_digest(tagged_image_url: str) -> str:
  inspect_command = [
      'docker',
      'image',
      'inspect',
      tagged_image_url,
      '-f',
      '{{index .RepoDigests 0}}',
  ]
  try:
    completed_process = utils.run_command(
        inspect_command,
        dry_run=False,  # Run even if --dry_run is True.
        capture_output=True,
        timeout=10)
  except subprocess.CalledProcessError as error:
    raise RuntimeError(f'Computing the repository digest for {tagged_image_url}'
                       ' failed. Has it been pushed to GCR?') from error
  _, repo_digest = completed_process.stdout.strip().split('@')
  return repo_digest


def update_rbe_reference(digest: str, dry_run: bool = False):
  print('Updating WORKSPACE file for rbe-toolchain')
  digest_updates = 0
  for line in fileinput.input(files=['WORKSPACE'], inplace=True):
    if line.strip().startswith('digest ='):
      digest_updates += 1
      if dry_run:
        print(line, end='')
      else:
        print(re.sub(DIGEST_REGEX, digest, line), end='')
    else:
      print(line, end='')

  if digest_updates > 1:
    raise RuntimeError(
        "There is more than one instance of 'digest =' in the WORKSPACE file. "
        "This means that more than just the 'rbe_toolchain' digest was "
        "overwritten, and the file should be restored.")


def update_references(image_url: str, digest: str, dry_run: bool = False):
  """Updates all references to 'image_url' with a sha256 digest."""
  print(f'Updating references to {image_url}')

  grep_command = ['git', 'grep', '-l', f'{image_url}@sha256']
  try:
    completed_process = run_command(grep_command,
                                    capture_output=True,
                                    timeout=5)
  except subprocess.CalledProcessError as error:
    if error.returncode == 1:
      print(f'Found no references to {image_url}')
      return
    raise error

  # Update references in all grepped files.
  files = completed_process.stdout.split()
  print(f'Updating references in {len(files)} files: {files}')
  if not dry_run:
    for line in fileinput.input(files=files, inplace=True):
      print(re.sub(f'{image_url}@{DIGEST_REGEX}', f'{image_url}@{digest}',
                   line),
            end='')


# TODO typing
def get_prod_digests() -> Dict[str, str]:
  with open(utils.PROD_DIGESTS_PATH, "r") as f:
    images_with_digests = {}
    for line in f:
      url, digest = line.strip().split("@")
      images_with_digests[url] = digest
  return images_with_digests


if __name__ == '__main__':
  args = parse_arguments()

  # Ensure the user has the correct authorization to push to GCR.
  utils.check_gcloud_auth(dry_run=args.dry_run)

  images_to_process = get_ordered_images_to_process(args.images)
  print(f'Also processing dependent images. Will process: {images_to_process}')

  dependencies = get_dependencies(images_to_process)
  print(f'Pulling image dependencies: {list(dependencies)}')
  prod_digests = get_prod_digests()
  for dependency in dependencies:
    if dependency in prod_digests:
      image_with_digest = f'{dependency}@{prod_digests[dependency]}'
      utils.run_command(["docker", "pull", image_with_digest],
                        dry_run=args.dry_run)

  for image in images_to_process:
    print(f'Processing image {image}')
    image_url = posixpath.join(IREE_GCR_URL, image)
    tagged_image_url = f'{image_url}'
    image_path = os.path.join(DOCKER_DIR, image)

    utils.run_command(
        ['docker', 'build', '--tag', tagged_image_url, image_path],
        dry_run=args.dry_run)

    utils.run_command(['docker', 'push', tagged_image_url],
                      dry_run=args.dry_run)

    digest = get_repo_digest(tagged_image_url)

    # Check that the image is in 'prod_digests.txt' and append it to the list
    # in the file if it isn't.
    if image_url not in prod_digests:
      image_with_digest = f'{image_url}@{digest}'
      print(
          f'Adding new image {image_with_digest} to {utils.PROD_DIGESTS_PATH}')
      if not args.dry_run:
        with open(utils.PROD_DIGESTS_PATH, 'a') as f:
          f.write(f'{image_with_digest}\n')

    # Just hardcode this oddity
    if image == 'rbe-toolchain':
      update_rbe_reference(digest, dry_run=args.dry_run)
    update_references(image_url, digest, dry_run=args.dry_run)
