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
import functools
import os
import subprocess

IREE_GCR_URL = 'gcr.io/iree-oss/'
DOCKER_DIR = 'build_tools/docker/'

# Map from image names to images that depend on them.
IMAGES_TO_DEPENDENT_IMAGES = {
    'bazel': ['bazel-bindings'],
    'bazel-bindings': ['bazel-tensorflow'],
    'bazel-tensorflow': [],
    'cmake': ['cmake-android', 'cmake-nvidia'],
    'cmake-android': [],
    'cmake-nvidia': [],
    'rbe-toolchain': [],
}

IMAGES_HELP = [f'`{name}`' for name in IMAGES_TO_DEPENDENT_IMAGES.keys()]
IMAGES_HELP = f'{", ".join(IMAGES_HELP[:-1])} or {IMAGES_HELP[-1]}'

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
      dest='images',
      type=str,
      required=True,
      action='append',
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
  for image in args.images:
    if image == "all":
      args.images = IMAGES_TO_DEPENDENT_IMAGES.keys()
    elif image not in IMAGES_TO_DEPENDENT_IMAGES.keys():
      raise parser.error('Expected --image to be one of:\n'
                         f'  {IMAGES_HELP}\n'
                         f'but got `{image}`.')
  return args


def cmp_images_by_dependency(image1, image2):
  if image2 in IMAGES_TO_DEPENDENT_IMAGES[image1]:
    return -1
  if image1 in IMAGES_TO_DEPENDENT_IMAGES[image2]:
    return 1
  return (image1 > image2) - (image1 < image2)


if __name__ == '__main__':
  args = parse_arguments()

  # Ensure the user has the correct authorization if they try to push to GCR.
  if args.push:
    subprocess.check_output(['gcloud', 'auth', 'configure-docker'])

  # Check if any images depend on `args.images` and update them if they do.
  images_to_update_set = set(args.images)
  for image in args.images:
    images_to_update_set.update(IMAGES_TO_DEPENDENT_IMAGES[image])

  # Topo sort by image dependency
  images_to_update = sorted(
      images_to_update_set, key=functools.cmp_to_key(cmp_images_by_dependency))

  for image in images_to_update:
    print(f'Updating image {image}')
    image_url = os.path.join(IREE_GCR_URL, f'{image}:{args.tag}')
    image_path = os.path.join(DOCKER_DIR, image.replace('-', '_'))
    subprocess.check_output(
        ['docker', 'build', '--tag', image_url, image_path])
    if args.push:
      subprocess.check_output(['docker', 'push', image_url])

  if 'rbe-toolchain' in images_to_update:
    print(RBE_MESSAGE)
