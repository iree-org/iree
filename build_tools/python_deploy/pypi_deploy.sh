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
#   python -m venv .venv
#   source .venv/bin/activate
#   python -m pip install -r ./pypi_deploy_requirements.txt
#   ./pypi_deploy.sh iree-3.4.0rc20250430            # dry run
#   ./pypi_deploy.sh --publish iree-3.4.0rc20250430  # upload to PyPI
#
# By default the script runs as a dry run: wheels are downloaded, promoted and
# validated with `twine check`, but nothing is uploaded. Pass --publish to
# upload to PyPI.
#
# To publish the wheels from a previous dry run without downloading them again,
# pass the directory that the dry run printed:
#   ./pypi_deploy.sh --publish --wheel-dir /tmp/iree_pypi_wheels.XXXXX
#
# Each package is uploaded separately. If an upload fails (e.g. because your
# credentials lack permission for that PyPI project), the remaining packages
# are still uploaded and the failures are listed in a summary at the end.

set -euo pipefail

function print_usage() {
  echo "Usage: $0 [--publish] <release tag, e.g. iree-3.4.0rc20250430>"
  echo "       $0 [--publish] --wheel-dir <directory from a previous run>"
}

PUBLISH=0
RELEASE=""
WHEEL_DIR=""
while (( $# > 0 )); do
  case "$1" in
    --publish)
      PUBLISH=1
      ;;
    --wheel-dir)
      if (( $# < 2 )); then
        echo "--wheel-dir requires a directory."
        print_usage
        exit 1
      fi
      WHEEL_DIR="$2"
      shift
      ;;
    --wheel-dir=*)
      WHEEL_DIR="${1#--wheel-dir=}"
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
    *)
      if [[ -n "${RELEASE}" ]]; then
        echo "Only one release may be given."
        print_usage
        exit 1
      fi
      RELEASE="$1"
      ;;
  esac
  shift
done
if [[ -n "${WHEEL_DIR}" && -n "${RELEASE}" ]]; then
  echo "Pass either a release tag or --wheel-dir, not both."
  print_usage
  exit 1
fi
if [[ -z "${WHEEL_DIR}" && -z "${RELEASE}" ]]; then
  print_usage
  exit 1
fi
if [[ -n "${WHEEL_DIR}" && ! -d "${WHEEL_DIR}" ]]; then
  echo "Wheel directory '${WHEEL_DIR}' does not exist."
  exit 1
fi

# Packages that every release is expected to contain, as named in wheel files.
EXPECTED_PACKAGES=(
  iree_base_compiler
  iree_base_runtime
  iree_tools_tf
  iree_tools_tflite
)

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
REQUIREMENTS_FILE="${SCRIPT_DIR}/pypi_deploy_requirements.txt"
if [[ -n "${WHEEL_DIR}" ]]; then
  TMPDIR="$(readlink -f -- "${WHEEL_DIR}")"
else
  TMPDIR="$(mktemp --directory --tmpdir iree_pypi_wheels.XXXXX)"
fi

function check_command_exists() {
  if ! command -v "$1" > /dev/null; then
    echo "$1 not found."
    return 1
  fi
  return 0
}

function check_python_package_installed() {
  if ! pip show "$1" > /dev/null; then
    echo "$1 not installed."
    return 1
  fi
  return 0
}

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
    if ! check_python_package_installed "${package}"; then
      echo "Recommend installing python dependencies in a venv using pypi_deploy_requirements.txt"
      exit 1
    fi

  done < <(cat "${REQUIREMENTS_FILE}")
}

function download_wheels() {
  echo ""
  echo "Downloading wheels from '${RELEASE}'..."
  gh release download "${RELEASE}" --repo iree-org/iree --pattern "*.whl"

  echo ""
  echo "Downloaded wheels:"
  ls
}

function confirm_or_exit() {
  local reply=""
  if ! read -r -p "$1 [y/N] " reply < /dev/tty; then
    echo ""
    echo "Could not read confirmation, aborting."
    exit 1
  fi
  if [[ ! "${reply}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Aborting."
    exit 1
  fi
}

function check_expected_packages() {
  echo ""
  echo "Checking for expected packages..."
  local missing=()
  local package
  for package in "${EXPECTED_PACKAGES[@]}"; do
    local wheels=( "${package}"-*.whl )
    if [[ -e "${wheels[0]}" ]]; then
      echo "  ${package}: ${#wheels[@]} wheel(s)"
    else
      echo "  ${package}: MISSING"
      missing+=( "${package}" )
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo ""
    echo "'${RELEASE:-${TMPDIR}}' has no wheels for: ${missing[*]}"
    confirm_or_exit "Continue without them?"
  fi
}

function edit_release_versions() {
  echo ""
  echo "Editing release versions..."
  for file in *
  do
    ${SCRIPT_DIR}/promote_whl_from_rc_to_final.py ${file} --delete-old-wheel
  done

  echo "Edited wheels:"
  ls
}

function upload_wheels() {
  # In a dry run, `twine check` stands in for `twine upload` so the wheels are
  # validated.
  local twine_args=( upload --verbose )
  local action="Uploading"
  local done_label="uploaded"
  echo ""
  if (( PUBLISH )); then
    echo "Uploading wheels..."
  else
    twine_args=( check --strict )
    action="Checking"
    done_label="ok"
    echo "Dry run: checking wheels instead of uploading (pass --publish to upload)..."
  fi

  # Group and upload wheels by package.
  local packages=()
  local file
  for file in *.whl; do
    packages+=( "${file%%-*}" )
  done
  mapfile -t packages < <(printf '%s\n' "${packages[@]}" | sort -u)

  local succeeded=()
  local failed=()
  local package
  for package in "${packages[@]}"; do
    echo ""
    echo "${action} ${package}..."
    if twine "${twine_args[@]}" "${package}"-*.whl; then
      succeeded+=( "${package}" )
    else
      echo "${action} ${package} failed, continuing with remaining packages."
      failed+=( "${package}" )
    fi
  done

  echo ""
  if (( PUBLISH )); then
    echo "Upload summary:"
  else
    echo "Dry run. Wheels that would be uploaded:"
    ls -1
    echo ""
    echo "Check summary:"
  fi
  for package in "${succeeded[@]}"; do
    echo "  ${done_label}: ${package}"
  done
  for package in "${failed[@]}"; do
    echo "  FAILED: ${package}"
  done
  if (( PUBLISH && ${#failed[@]} > 0 )); then
    echo ""
    echo "Some packages were not uploaded. Their wheels are kept in ${TMPDIR}."
    echo "  twine upload ${TMPDIR}/<package>-*.whl"
  fi
  if (( ! PUBLISH )); then
    echo ""
    echo "To upload these wheels without downloading them again, run:"
    echo "  $0 --publish --wheel-dir ${TMPDIR}"
  fi
}


function main() {
  local source="${RELEASE:-${TMPDIR}}"
  if (( PUBLISH )); then
    echo "Publishing '${source}' to PyPI."
  else
    echo "Dry run for '${source}', nothing will be uploaded (pass --publish to upload)."
  fi
  echo "Changing into ${TMPDIR}"
  cd "${TMPDIR}"

  set +e
  check_requirements

  if [[ -z "${WHEEL_DIR}" ]] && ! check_command_exists gh; then
    echo "The GitHub CLI 'gh' is required. See https://github.com/cli/cli#installation."
    echo " Googlers, the PPA should already be on your linux machine."
    exit 1
  fi
  set -e

  if [[ -z "${WHEEL_DIR}" ]]; then
    download_wheels
  else
    echo ""
    echo "Using wheels from '${TMPDIR}' instead of downloading:"
    ls
  fi
  check_expected_packages
  edit_release_versions
  upload_wheels
}

main
