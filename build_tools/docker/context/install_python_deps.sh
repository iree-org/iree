#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! [[ -f build_requirements.txt ]]; then
  echo "Couldn't find build_requirements.txt file in current directory" >&2
  exit 1
fi

PYTHON_VERSION="$1"

apt-get update

apt-get install -y \
  "python${PYTHON_VERSION}" \
  "python${PYTHON_VERSION}-dev"

update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1

apt-get install -y \
  python3-pip \
  python3-setuptools \
  python3-distutils \
  python3-venv \
  "python${PYTHON_VERSION}-venv"

# Note that we use --ignore-installed when installing packages that may have
# been auto-installed by the OS package manager (i.e. PyYAML is often an
# implicit OS-level dep). This should not break so long as we do not
# subsequently reinstall it on the OS side. Failing to do this will yield a
# hard error with pip along the lines of:
#   Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we
#   cannot accurately determine which files belong to it which would lead to
#   only a partial uninstall.
python3 -m pip install --ignore-installed --upgrade -r build_requirements.txt
