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

  # .md files for the website.
  "./docs/website/**/*.md"

  # Some developer documentation .md files that we may move to the website.
  "./docs/developers/developing_iree/*.md"
)

declare -a excluded_files_patterns=(
  "**/third_party/**"
  "**/node_modules/**"

  # Exclude generated files.
  "./docs/website/docs/reference/mlir-dialects/!(index).md"
)

# ${excluded_files_patterns} is expanded into
# "--ignore=pattern1 --ignore=pattern2 ...", since markdownlint doesn't accept
# "--ignore pattern1 pattern2 ...".
# The expansion trick is explained in
# https://stackoverflow.com/questions/20366609/prefix-and-postfix-elements-of-a-bash-array
markdownlint "${included_files_patterns[@]}" \
    --config=./docs/.markdownlint.yml \
    "${excluded_files_patterns[@]/#/--ignore=}"
