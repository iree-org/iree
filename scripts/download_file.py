#!/usr/bin/env python3

# Copyright 2021 Google LLC
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
"""Downloads a file from the web and untars it if necessary."""

import argparse
import os
import requests
import tarfile


def parse_arguments():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("source_url",
                      type=str,
                      metavar="<source-url>",
                      help="Source URL to download")
  parser.add_argument("-o",
                      "--output",
                      type=str,
                      required=True,
                      metavar="<output-directory>",
                      help="Output directory to contain the file")
  return parser.parse_args()


def main(args):
  name = args.source_url.split("/")[-1]

  if not os.path.isdir(args.output):
    os.makedirs(args.output)
  output_file = os.path.join(args.output, name)

  response = requests.get(args.source_url)
  if response.status_code != 200:
    raise requests.RequestException(
        f"Failed to download file with status code {response.status_code}")

  with open(output_file, "wb") as f:
    f.write(response.content)

  if name.endswith("tar.gz") or name.endswith("tgz"):
    with tarfile.open(output_file, "r") as f:
      f.extractall(args.output)


if __name__ == "__main__":
  main(parse_arguments())
