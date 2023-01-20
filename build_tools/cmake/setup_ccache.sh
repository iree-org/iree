# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ccache (https://ccache.dev/) setup, with read/write + local/remote options.
#
# Defaults to only reading from the shared remote cache (hosted on GCS) used by
# our Linux CI. The postsubmit CI writes to the cache, for presubmit CI and
# local builds to read from.
#
# Local caches can also be used to interface with external remote caches
# (like https://github.com/actions/cache) by
#   1. downloading the cache directory
#   2. sourcing this file with IREE_USE_LOCAL_CCACHE=1 IREE_READ_REMOTE_CCACHE=0
#   3. building with CMake
#   4. uploading the cache directory (optionally)
#
# Note: this file must be *sourced* not executed.

set -euo pipefail

# Configuration environment variables.
IREE_READ_REMOTE_CCACHE="${IREE_READ_REMOTE_CCACHE:-1}"
IREE_WRITE_REMOTE_CCACHE="${IREE_WRITE_REMOTE_CCACHE:-0}"
IREE_USE_LOCAL_CCACHE="${IREE_USE_LOCAL_CCACHE:-0}"

if (( ${IREE_WRITE_REMOTE_CCACHE} == 1 && ${IREE_READ_REMOTE_CCACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_CCACHE' (${IREE_WRITE_REMOTE_CCACHE})" \
       " set without 'IREE_READ_REMOTE_CCACHE' (${IREE_READ_REMOTE_CCACHE})"
fi
if (( ${IREE_USE_LOCAL_CCACHE} == 1 && ${IREE_READ_REMOTE_CCACHE} == 1)); then
  echo "Can't have 'IREE_USE_LOCAL_CCACHE' (${IREE_USE_LOCAL_CCACHE})" \
       " set with 'IREE_READ_REMOTE_CCACHE' (${IREE_READ_REMOTE_CCACHE})"
fi

if (( IREE_READ_REMOTE_CCACHE == 1 || IREE_USE_LOCAL_CCACHE == 1 )); then
  export IREE_USE_CCACHE=1
  export CMAKE_C_COMPILER_LAUNCHER=$(which ccache)
  export CMAKE_CXX_COMPILER_LAUNCHER=$(which ccache)
  $(which ccache) --zero-stats
else
  export IREE_USE_CCACHE=0
fi

if (( IREE_READ_REMOTE_CCACHE == 1 )); then
  export CCACHE_REMOTE_STORAGE="http://storage.googleapis.com/iree-sccache/ccache"
  export CCACHE_REMOTE_ONLY=1
  if (( IREE_WRITE_REMOTE_CCACHE == 1 )); then
    set +x # Don't leak the token (even though it's short-lived)
    export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|bearer-token=${IREE_CCACHE_GCP_TOKEN}"
    set -x
  else
    export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|read-only"
  fi
fi
