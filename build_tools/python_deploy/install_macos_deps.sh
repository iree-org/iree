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

# These can be discovered at https://www.python.org/downloads/macos/
# Each spec is <version>@<installer SHA-256>@<installer URL>.
PYTHON_SPECS=(
  3.13@67c6f0a3190851e0013214d5abd725a42ec398ff1b50eec47826820fd052d86b@https://www.python.org/ftp/python/3.13.1/python-3.13.1-macos11.pkg
  3.12@c411b5372d563532f5e6b589af7eb16e95613d61bd5af7bfe78563467130bbff@https://www.python.org/ftp/python/3.12.8/python-3.12.8-macos11.pkg
  3.11@b6cfdee2571ca56ee895043ca1e7110fb78a878cee3eb0c21accb2de34d24b55@https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg
)

# The python.org installers are signed with this Apple Developer ID.
PYTHON_SIGNER="Developer ID Installer: Python Software Foundation (BMM5U3QVKW)"

for python_spec in "${PYTHON_SPECS[@]}"; do
  python_version="${python_spec%%@*}"
  url="${python_spec##*@}"
  sha256="${python_spec#*@}"
  sha256="${sha256%%@*}"
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
    echo "$sha256  $download_path" | shasum -a 256 -c -

    echo "Verifying the signature of $download_path"
    signature="$(pkgutil --check-signature "$download_path")"
    echo "$signature"
    if [[ "$signature" != *"signed by a developer certificate issued by Apple"* ||
          "$signature" != *"$PYTHON_SIGNER"* ]]; then
      echo "ERROR: $download_path is not signed by '$PYTHON_SIGNER'"
      exit 1
    fi

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
