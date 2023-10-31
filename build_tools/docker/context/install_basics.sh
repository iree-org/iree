#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

# Useful utilities for building child images. Best practices would tell us to
# use multi-stage builds
# (https://docs.docker.com/develop/develop-images/multistage-build/) but it
# turns out that Dockerfile is a thoroughly non-composable awful format and that
# doesn't actually work that well. These deps are pretty small.
apt-get update && apt-get install -y git unzip wget curl gnupg2 lsb-release
