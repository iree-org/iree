#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# For deploying to PyPI, you will need to have credentials set up.
# Googlers can access the shared releasing account "google-iree-pypi-deploy"
# at http://go/iree-pypi-password
#
# Typical usage is to use keyring or create a ~/.pypirc file with:
#
#   [pypi]
#   username = __token__
#   password = <<API TOKEN>>
#
# You must have `gh` installed and authenticated (run `gh auth`).
#
# Usage:
# ./pypi_deploy.sh candidate-20220930.282

set -euo pipefail

RELEASE="$1"

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
REQUIREMENTS_FILE="${SCRIPT_DIR}/pypi_deploy_requirements.txt"
TMPDIR="$(mktemp --directory --tmpdir iree_pypi_wheels.XXXXX)"

function check_exists() {
  if ! command -v "$1" > /dev/null; then
    echo "$1 not found."
    exit 1
  fi
}

# It really *seems* like there should be a pip command to do this, but there's
# not, apparently.
function check_requirements() {
  while read line; do
    # Read in the package, ignoring everything after the first '='
    ret=0
    read -rd '=' package <<< "${line}" || ret=$?
    # exit code 1 means EOF (i.e. no '='), which is fine.
    if (( ret!=0 && ret!=1 )); then
      echo "Reading requirements file '${REQUIREMENTS_FILE}' failed."
      exit "${ret}"
    fi
    if ! check_exists "${package}"; then
      echo "Recommend installing python dependencies in a venv using pypi_deploy_requirements.txt"
      exit 1
    fi

  done < <(cat "${REQUIREMENTS_FILE}")
}

function download_wheels() {
  echo ""
  echo "Downloading wheels from '${RELEASE}'"
  gh release download "${RELEASE}" --repo openxla/iree --pattern "*.whl"
}

function upload_wheels() {
  twine upload --verbose *
}


function main() {
  echo "Changing into ${TMPDIR}"
  cd "${TMPDIR}"
  check_requirements

  if ! check_exists gh; then
    echo "The GitHub CLI 'gh' is required. See https://github.com/cli/cli#installation."
    echo " Googlers, the PPA should already be on your linux machine."
    exit 1
  fi

  download_wheels
  upload_wheels
}

main
