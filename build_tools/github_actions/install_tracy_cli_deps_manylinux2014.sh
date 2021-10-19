#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Installs deps on a manylinux2014 CentOS docker container needed for
# building Tracy CLI capture tool.

set -e

td="$(cd $(dirname $0) && pwd)"
yum -y install capstone-devel libzstd-devel
$td/install_tbb_manylinux2014.sh
