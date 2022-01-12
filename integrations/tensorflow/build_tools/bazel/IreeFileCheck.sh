#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A wrapper around FileCheck that sets the flags we always want set.

set -e

FileCheck --enable-var-scope --dump-input=fail "$@"
