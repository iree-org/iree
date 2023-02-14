#!/bin/zsh
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs dependencies on MacOS necessary to build IREE.
# Additional dependencies (i.e MoltenVK) may be needed to use all functionality.
#
# Usage:
#   sudo install_macos_deps.sh

set -e -o pipefail

if [[ "$(whoami)" != "root" ]]; then
  echo "ERROR: Must setup deps as root"
  exit 1
fi

PYTHON_INSTALLER_URLS=(
  "https://www.python.org/ftp/python/3.11.2/python-3.11.2-macos11.pkg"
  "https://www.python.org/ftp/python/3.10.10/python-3.10.10-macos11.pkg"
  "https://www.python.org/ftp/python/3.9.13/python-3.9.13-macos11.pkg"
)

PYTHON_SPECS=(
  3.11@https://www.python.org/ftp/python/3.11.2/python-3.11.2-macos11.pkg
  3.10@https://www.python.org/ftp/python/3.10.5/python-3.10.5-macos11.pkg
  3.9@https://www.python.org/ftp/python/3.9.13/python-3.9.13-macos11.pkg
)

for python_spec in $PYTHON_SPECS; do
  python_version="${python_spec%%@*}"
  url="${python_spec##*@}"
  echo "-- Installing Python $python_version from $url"
  python_path="/Library/Frameworks/Python.framework/Versions/$python_version"
  python_exe="$python_path/bin/python3"

  # Install Python.
  if ! [ -x "$python_exe" ]; then
    package_basename="$(basename $url)"
    download_path="/tmp/iree_python_install/$package_basename"
    mkdir -p "$(dirname $download_path)"
    echo "Downloading $url -> $download_path"
    curl $url -o "$download_path"

    echo "Installing $download_path"
    installer -pkg "$download_path" -target /
  else
    echo ":: Python version already installed. Not reinstalling."
  fi

  echo ":: Python version $python_version installed:"
  $python_exe --version
  $python_exe -m pip --version

  echo ":: Installing system pip packages"
  $python_exe -m pip install --upgrade pip
  $python_exe -m pip install --upgrade delocate
done

echo "*** All done ***"
