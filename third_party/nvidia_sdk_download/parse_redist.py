#!/usr/bin/env python3
# Copyright 2021, NVIDIA Corporation
# SPDX-License-Identifier: MIT
"""
Sample parser for redistrib JSON manifests
1. Downloads each archive
2. Validates SHA256 checksums
3. Extracts archives
4. Flattens into a collapsed directory structure

Forked from https://github.com/NVIDIA/build-system-archive-import-examples/blob/355e25cca11725e88984443a6a343dffeb43308a/parse_redist.py
and patched:
  - avoid a dependency on the non-standard requests package (see
the http_get helper) by using urllib directly
  - explicit error handling on hash mismatch
  - always download, even if files exist
  - remove dependence on deprecated distutils copy_tree in favor of shutil
"""
import argparse
import os.path
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
import sys
import urllib.request

__version__ = "0.1.0"

ARCHIVES = {}
DOMAIN = "https://developer.download.nvidia.com"
OUTPUT = "flat"
PRODUCT = None
LABEL = None
URL = None
OS = None
ARCH = None
PLATFORM = None
COMPONENT = None

# Default actions
RETRIEVE = True
VALIDATE = True
UNROLLED = True
COLLAPSE = True


def http_get(url):
  """Fetch the contents of a URL."""
  with urllib.request.urlopen(url) as f:
    data = f.read()
    if hasattr(f, "status"):
      # For >= 3.9
      status_code = f.status
    else:
      # Deprecated in 3.9
      statuc_code = f.code
    if status_code != 200:
      raise IOError("  -> Failed to download: " + url)
    return data


def err(msg):
  """Print error message and exit"""
  print("ERROR: " + msg)
  sys.exit(1)


def fetch_file(full_path, filename):
  """Download file to disk"""
  print(":: Fetching: " + full_path)
  download_data = http_get(full_path)
  with open(filename, "wb") as file:
    file.write(download_data)
    print("  -> Wrote: " + filename)


def get_hash(filename):
  """Calculate SHA256 checksum for file"""
  buffer_size = 65536
  sha256 = hashlib.sha256()
  with open(filename, "rb") as file:
    while True:
      chunk = file.read(buffer_size)
      if not chunk:
        break
      sha256.update(chunk)
  return sha256.hexdigest()


def check_hash(filename, checksum):
  """Compare checksum with expected"""
  sha256 = get_hash(filename)
  if checksum == sha256:
    print("     Verified sha256sum: " + sha256)
  else:
    raise IOError(f"Mismatch sha256sum: Calculation={sha256}, "
                  f"Expectation={checksum} for {filename}")


def flatten_tree(src, dest):
  """Merge hierarchy from multiple directories"""

  # Should use shutil.copytree(dirs_exist_ok=True), but that isn't available
  # until Python 3.8.
  def copytree(src, dst):
    if not os.path.exists(dst):
      os.makedirs(dst)
    for item in os.listdir(src):
      s = os.path.join(src, item)
      d = os.path.join(dst, item)
      if os.path.isdir(s):
        copytree(s, d)
      else:
        if not os.path.exists(d):
          shutil.copy2(s, d)

  try:
    copytree(src, dest)
  finally:
    shutil.rmtree(src)


def fetch_action(parent):
  """Do actions while parsing JSON"""
  for component in MANIFEST.keys():
    if not 'name' in MANIFEST[component]:
      continue

    if COMPONENT is not None and component != COMPONENT:
      continue

    print("\n" + MANIFEST[component]['name'] + ": " +
          MANIFEST[component]['version'])

    for platform in MANIFEST[component].keys():
      if not platform in ARCHIVES:
        ARCHIVES[platform] = []

      if not isinstance(MANIFEST[component][platform], str):
        if PLATFORM is not None and platform != PLATFORM:
          print("  -> Skipping platform: " + platform)
          continue

        full_path = parent + MANIFEST[component][platform]['relative_path']
        filename = os.path.basename(full_path)
        ARCHIVES[platform].append(filename)

        if RETRIEVE:
          # Download archive
          fetch_file(full_path, filename)
        elif os.path.exists(filename):
          print("  -> Found: " + filename)

        checksum = MANIFEST[component][platform]['sha256']
        if VALIDATE and os.path.exists(filename):
          # Compare checksum
          check_hash(filename, checksum)


def post_action():
  """Extract archives and merge directories"""
  if len(ARCHIVES) == 0:
    return

  print("\nArchives:")
  if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

  for platform in ARCHIVES:
    for archive in ARCHIVES[platform]:
      # Tar files
      if UNROLLED and re.search(r"\.tar\.", archive):
        print(":: tar: " + archive)
        tarball = tarfile.open(archive)
        topdir = os.path.commonprefix(tarball.getnames())
        tarball.extractall()
        tarball.close()
        print("  -> Extracted: " + topdir + "/")
        if COLLAPSE:
          flatten_tree(topdir, OUTPUT + "/" + platform)

      # Zip files
      elif UNROLLED and re.search(r"\.zip", archive):
        print(":: zip: " + archive)
        with zipfile.ZipFile(archive) as zippy:
          topdir = os.path.commonprefix(zippy.namelist())
          zippy.extractall()
        zippy.close()

        print("  -> Extracted: " + topdir)
        if COLLAPSE:
          flatten_tree(topdir, OUTPUT + "/" + platform)

  print("\nOutput: " + OUTPUT + "/")
  for item in sorted(os.listdir(OUTPUT)):
    if os.path.isdir(OUTPUT + "/" + item):
      print(" - " + item + "/")
    elif os.path.isfile(OUTPUT + "/" + item):
      print(" - " + item)


# If running standalone
if __name__ == '__main__':
  # Parse CLI arguments
  PARSER = argparse.ArgumentParser()
  # Input options
  PARSER_GROUP = PARSER.add_mutually_exclusive_group(required=True)
  PARSER_GROUP.add_argument('-u', '--url', dest='url', help='URL to manifest')
  PARSER_GROUP.add_argument('-l',
                            '--label',
                            dest='label',
                            help='Release label version')
  PARSER.add_argument('-p', '--product', dest='product', help='Product name')
  PARSER.add_argument('-o', '--output', dest='output', help='Output directory')
  # Filter options
  PARSER.add_argument('--component', dest='component', help='Component name')
  PARSER.add_argument('--os', dest='os', help='Operating System')
  PARSER.add_argument('--arch', dest='arch', help='Architecture')
  # Toggle actions
  PARSER.add_argument('-w', '--download', dest='retrieve', action='store_true', \
       help='Download archives', default=True)
  PARSER.add_argument('-W', '--no-download', dest='retrieve', action='store_false', \
       help='Parse manifest without downloads')
  PARSER.add_argument('-s', '--checksum', dest='validate', action='store_true', \
       help='Verify SHA256 checksum', default=True)
  PARSER.add_argument('-S', '--no-checksum', dest='validate', action='store_false', \
       help='Skip SHA256 checksum validation')
  PARSER.add_argument('-x', '--extract', dest='unrolled', action='store_true', \
       help='Extract archives', default=True)
  PARSER.add_argument('-X', '--no-extract', dest='unrolled', action='store_false', \
       help='Do not extract archives')
  PARSER.add_argument('-f', '--flatten', dest='collapse', action='store_true', \
       help='Collapse directories', default=True)
  PARSER.add_argument('-F', '--no-flatten', dest='collapse', action='store_false', \
       help='Do not collapse directories')

  ARGS = PARSER.parse_args()
  #print(ARGS)
  RETRIEVE = ARGS.retrieve
  VALIDATE = ARGS.validate
  UNROLLED = ARGS.unrolled
  COLLAPSE = ARGS.collapse

  # Define variables
  if ARGS.label is not None:
    LABEL = ARGS.label
  if ARGS.product is not None:
    PRODUCT = ARGS.product
  if ARGS.url is not None:
    URL = ARGS.url
  if ARGS.output is not None:
    OUTPUT = ARGS.output
  if ARGS.component is not None:
    COMPONENT = ARGS.component
  if ARGS.os is not None:
    OS = ARGS.os
  if ARGS.arch is not None:
    ARCH = ARGS.arch

#
# Setup
#

# Sanity check
if not UNROLLED:
  COLLAPSE = False

# Short-hand
if LABEL:
  if PRODUCT:
    URL = f"{DOMAIN}/compute/{PRODUCT}/redist/redistrib_{LABEL}.json"
  else:
    err("Must pass --product argument")

# Concatentate
if ARCH is not None and OS is not None:
  PLATFORM = f"{OS}-{ARCH}"
elif ARCH is not None and OS is None:
  err("Must pass --os argument")
elif OS is not None and ARCH is None:
  err("Must pass --arch argument")

#
# Run
#

# Parse JSON
try:
  MANIFEST = json.loads(http_get(URL))
except json.decoder.JSONDecodeError:
  err("redistrib JSON manifest file not found")

print(":: Parsing JSON: " + URL)

# Do stuff
fetch_action(os.path.dirname(URL) + "/")
post_action()

### END ###
