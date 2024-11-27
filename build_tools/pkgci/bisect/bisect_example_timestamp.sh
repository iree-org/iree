#!/bin/bash
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple example script to test bisect with.
# Fails for commits after Nov 19, 2024.

# Example usage with https://git-scm.com/docs/git-bisect:
#
#   git bisect start --no-checkout --first-parent
#   git bisect good iree-3.0.0
#   git bisect bad iree-3.1.0rc20241122
#   git bisect run bisect_example_timestamp.sh
#
#     running  'bisect_example_timestamp.sh'
#     Commit 26ef79aa7c has timestamp: 1732059549
#     Timestamp >= 1732000000, exit 1
#     Bisecting: 10 revisions left to test after this (roughly 4 steps)
#     ...
#     5b0740c97a33edce29e753b14b9ff04789afcc53 is the first bad commit
#
# Example usage with ./bisect_packages.py (even though this doesn't use any
# release artifacts like `iree-compile`):
#
#   ./bisect_packages.py \
#     --good-ref=iree-3.0.0 \
#     --bad-ref=iree-3.1.0rc20241122 \
#     --test-script=./bisect_example_timestamp.sh

SHORT_HASH=$(git rev-parse --short BISECT_HEAD)
COMMIT_TIMESTAMP=$(git show --no-patch --format=%ct BISECT_HEAD)
echo "Commit ${SHORT_HASH} has timestamp: ${COMMIT_TIMESTAMP}"

if [ "$COMMIT_TIMESTAMP" -gt "1732000000" ]; then
    echo "  Timestamp >= 1732000000, exit 1"
    exit 1
else
    echo "  Timestamp < 1732000000, exit 0"
    exit 0
fi
