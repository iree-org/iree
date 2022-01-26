#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -ou pipefail

# Enable globstar so ** globs recursively
shopt -s globstar

yamllint --strict github/**/*.yml build_tools/buildkite/**/*.yml
exitcode=$?

shopt -u globstar
exit $exitcode
