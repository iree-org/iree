# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# sccache (https://github.com/mozilla/sccache/) setup, focused on Azure Blob
# Storage. See https://github.com/mozilla/sccache/blob/main/docs/Azure.md.
#
# If the `SCCACHE_AZURE_CONNECTION_STRING` environment variable is set, this
# will enable sccache. Note that `SCCACHE_AZURE_BLOB_CONTAINER` should also be
# set. The `SCCACHE_CACHE_ZSTD_LEVEL` and `SCCACHE_AZURE_KEY_PREFIX`
# environment variables are also recommended. We could give them default values
# here if we wanted.
#
# If the `SCCACHE_AZURE_CONNECTION_STRING` environment variable is _not_ set,
# this keeps sccache disabled. It does _not_ use a readonly cache.
#
# Note: this file must be *sourced* not executed.

set -eo pipefail

if [ -n "$SCCACHE_AZURE_CONNECTION_STRING" ]; then
  echo "Connection string set, using sccache"
  export IREE_USE_SCCACHE=1
  export CMAKE_C_COMPILER_LAUNCHER="$(which sccache)"
  export CMAKE_CXX_COMPILER_LAUNCHER="$(which sccache)"
else
  echo "Connection string _not_ set, skipping sccache setup"
  unset SCCACHE_AZURE_CONNECTION_STRING
  export IREE_USE_SCCACHE=0
fi

sccache --zero-stats
sccache --show-stats
