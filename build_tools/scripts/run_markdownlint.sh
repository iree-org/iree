#!/bin/bash

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs Markdownlint on Markdown (.md) files
# https://github.com/igorshubovych/markdownlint-cli

set -euo pipefail

declare -a included_files_patterns=(
  # All .md files (disabled while we decide how rigorously to apply lint checks)
  # "./**/*.md"

  # Just .md files for the user-facing website.
  "./docs/website/**/*.md"
)

declare -a excluded_files_patterns=(
  "/third_party/"
  "**/node_modules/**"
)

markdownlint "${included_files_patterns[*]}" \
    --config ./docs/.markdownlint.yml \
    --ignore "${excluded_files_patterns[*]}"
