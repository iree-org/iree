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

Example usage:

Rebuild the cmake image and all images that transitively on depend on it,
tagging them with `latest`:
  python3 build_tools/docker/manage_images.py --build --image cmake

Print out output for rebuilding the cmake image and all images that
transitively on depend on it, but don't take side-effecting actions:
  python3 build_tools/docker/manage_images.py --build --image cmake --dry-run

Push all `prod` images to GCR:
  python3 build_tools/docker/manage_images.py --push --tag prod --images all

Rebuild and push all images and update references to them in the repository:
  python3 build_tools/docker/manage_images.py --push --images all
  --update-references
"""

import argparse
import fileinput
import os
import posixpath
import re
import subprocess
import sys

IREE_GCR_URL = 'gcr.io/iree-oss/'
DOCKER_DIR = 'build_tools/docker/'

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
      '--tag',
      type=str,
      default='latest',
      help='Tag for the images to build. Defaults to `latest` (which is good '
      'for testing changes in a PR). Use `prod` to update the images that the '
      'CI caches.')
  parser.add_argument('--pull',
                      action='store_true',
                      help='Pull the specified image before building.')
  parser.add_argument('--build',
                      action='store_true',
                      help='Build new images from the current Dockerfiles.')
  parser.add_argument(
      '--push',
      action='store_true',
      help='Push the built images to GCR. Requires gcloud authorization.')
  parser.add_argument(
      '--update_references',
      '--update-references',
      action='store_true',
      help='Update all references to the specified images to point at the new'
      ' digest.')
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


def get_ordered_images_to_process(images):
  unmarked_images = list(images)
  # Python doesn't have a builtin OrderedSet
  marked_images = set()
  order = []

  def visit(image):
    if image in marked_images:
      return
    for dependent_images in IMAGES_TO_DEPENDENT_IMAGES[image]:
      visit(dependent_images)
    marked_images.add(image)
    order.append(image)

  while unmarked_images:
    visit(unmarked_images.pop())

  order.reverse()
  return order


def stream_command(command, dry_run=False):
  print(f'Running: `{" ".join(command)}`')
  if dry_run:
    return 0
  process = subprocess.Popen(command,
                             bufsize=1,
                             stderr=subprocess.STDOUT,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
  for line in process.stdout:
    print(line, end='')

  if process.poll() is None:
    raise RuntimeError('Unexpected end of output while process is not finished')
  return process.poll()


def check_stream_command(command, dry_run=False):
  exit_code = stream_command(command, dry_run=dry_run)
  if exit_code != 0:
    print(f'Command failed with exit code {exit_code}: `{" ".join(command)}`')
    sys.exit(exit_code)


def get_repo_digest(image):
  inspect_command = [
      'docker',
      'image',
      'inspect',
      f'{image}',
      '-f',
      '{{index .RepoDigests 0}}',
  ]
  inspect_process = subprocess.run(inspect_command,
                                   universal_newlines=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   timeout=10)
  if inspect_process.returncode != 0:
    print(f'Computing the repository digest for {image} failed.'
          ' Has it been pushed to GCR?')
    print(f'Output from `{" ".join(inspect_command)}`:')
    print(inspect_process.stdout, end='')
    print(inspect_process.stderr, end='')
    sys.exit(inspect_process.returncode)
  _, repo_digest = inspect_process.stdout.strip().split('@')
  return repo_digest


def update_rbe_reference(digest, dry_run=False):
  print('Updating WORKSPACE file for rbe-toolchain')
  for line in fileinput.input(files=['WORKSPACE'], inplace=(not dry_run)):
    if line.strip().startswith('digest ='):
      print(re.sub('sha256:[a-zA-Z0-9]+', digest, line), end='')
    else:
      print(line, end='')


def update_references(image_name, digest, dry_run=False):
  print(f'Updating references to {image_name}')

  grep_command = ['git', 'grep', '-l', f'{image_name}@sha256']
  grep_process = subprocess.run(grep_command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=5,
                                universal_newlines=True)
  if grep_process.returncode > 1:
    print(f'{" ".join(grep_command)} '
          f'failed with exit code {grep_process.returncode}')
    sys.exit(grep_process.returncode)
  if grep_process.returncode == 1:
    print(f'Found no references to {image_name}')
    return

  files = grep_process.stdout.split()
  print(f'Updating references in {len(files)} files: {files}')
  for line in fileinput.input(files=files, inplace=(not dry_run)):
    print(re.sub(f'{image_name}@sha256:[a-zA-Z0-9]+', f'{image_name}@{digest}',
                 line),
          end='')


if __name__ == '__main__':
  args = parse_arguments()

  # Ensure the user has the correct authorization if they try to push to GCR.
  if args.push:
    if stream_command(['which', 'gcloud']) != 0:
      print('gcloud not found.'
            ' See https://cloud.google.com/sdk/install for installation.')
      sys.exit(1)
    check_stream_command(['gcloud', 'auth', 'configure-docker'],
                         dry_run=args.dry_run)

  images_to_process = get_ordered_images_to_process(args.images)
  print(f'Also processing dependent images. Will process: {images_to_process}')

  for image in images_to_process:
    print(f'Processing image {image}')
    image_name = posixpath.join(IREE_GCR_URL, image)
    image_tag = f'{image_name}:{args.tag}'
    image_path = os.path.join(DOCKER_DIR, image)

    if args.pull:
      check_stream_command(['docker', 'pull', image_tag], dry_run=args.dry_run)

    if args.build:
      check_stream_command(['docker', 'build', '--tag', image_tag, image_path],
                           dry_run=args.dry_run)

    if args.push:
      check_stream_command(['docker', 'push', image_tag], dry_run=args.dry_run)

    if args.update_references:
      digest = get_repo_digest(image_tag)
      # Just hardcode this oddity
      if image == 'rbe-toolchain':
        update_rbe_reference(digest, dry_run=args.dry_run)
      update_references(image_name, digest, dry_run=args.dry_run)
