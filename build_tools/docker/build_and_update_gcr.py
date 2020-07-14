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
"""Builds the specified Docker images and optionally pushes them to GCR.

Example usage:
  python3 build_tools/docker/build_and_update_gcr.py --image cmake
"""

import argparse
import os
import subprocess

IREE_GCR_URL = 'gcr.io/iree-oss/'
DOCKER_DIR = 'build_tools/docker/'

IMAGES = [
    'bazel', 'bazel-bindings', 'bazel-tensorflow', 'cmake', 'cmake-android',
    'rbe-toolchain'
]
IMAGES_HELP = [f'`{name}`' for name in IMAGES]
IMAGES_HELP = f'{", ".join(IMAGES_HELP[:-1])} or {IMAGES_HELP[-1]}'

# Map from image names to images that depend on them.
IMAGES_TO_DEPENDENT_IMAGES = {
    'bazel': ['bazel-bindings', 'bazel-tensorflow'],
    'cmake': ['cmake-android']
}

RBE_MESSAGE = """
Remember to update the `rbe_default` digest in the `WORKSPACE` file to reflect
the new digest for the container.

Use `docker images --digests` to view the digest."""


def parse_arguments():
  """Parses command-line options."""
  parser = argparse.ArgumentParser(
      description="Build IREE's Docker images and optionally push them to GCR.")
  parser.add_argument(
      '--image',
      type=str,
      required=True,
      help=f'Name of the image to build: {IMAGES_HELP}.')
  parser.add_argument(
      '--tag',
      type=str,
      default='latest',
      help='Tags for the images to build. Defaults to `latest` (which is good '
      'for testing changes in a PR). Use `prod` to update the images that the '
      'OSS CI uses.')
  parser.add_argument(
      '--push',
      action='store_true',
      help='Push the built images to GCR. Requires gcloud authorization.')

  args = parser.parse_args()
  if args.image not in IMAGES:
    raise parser.error('Expected --image to be one of:\n'
                       f'  {IMAGES_HELP}\n'
                       f'but got `{args.image}`.')

  return args


if __name__ == '__main__':
  args = parse_arguments()

  # Ensure the user has the correct authorization if they try to push to GCR.
  if args.push:
    subprocess.check_output(['gcloud', 'auth', 'configure-docker'])

  # Check if any images depend on `args.image` and update them if they do.
  images_to_update = [args.image]
  if args.image in IMAGES_TO_DEPENDENT_IMAGES:
    images_to_update.extend(IMAGES_TO_DEPENDENT_IMAGES[args.image])

  for image in images_to_update:
    print(f'Updating image {image}')
    image_url = os.path.join(IREE_GCR_URL, f'{image}:{args.tag}')
    image_path = os.path.join(DOCKER_DIR, image.replace('-', '_'))
    subprocess.check_output(['docker', 'build', '--tag', image_url, image_path])
    if args.push:
      subprocess.check_output(['docker', 'push', image_url])

  if 'rbe-toolchain' in images_to_update:
    print(RBE_MESSAGE)
