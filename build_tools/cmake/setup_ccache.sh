# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file must be *sourced* not executed.

set -euo pipefail

IREE_READ_REMOTE_CCACHE="${IREE_READ_REMOTE_CCACHE:-1}"
IREE_WRITE_REMOTE_CCACHE="${IREE_WRITE_REMOTE_CCACHE:-0}"
if (( ${IREE_WRITE_REMOTE_CCACHE} == 1 && ${IREE_READ_REMOTE_CCACHE} != 1 )); then
  echo "Can't have 'IREE_WRITE_REMOTE_CCACHE' (${IREE_WRITE_REMOTE_CCACHE})" \
       " set without 'IREE_READ_REMOTE_CCACHE' (${IREE_READ_REMOTE_CCACHE})"
fi

if (( IREE_READ_REMOTE_CCACHE == 1 )); then
  export CCACHE_REMOTE_STORAGE="http://storage.googleapis.com/iree-sccache/ccache"
  export CCACHE_REMOTE_ONLY=1
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi
if (( IREE_WRITE_REMOTE_CCACHE == 1 )); then
  set +x # Don't leak the token (even though it's short-lived)
  export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|bearer-token=${IREE_CCACHE_GCP_TOKEN}"
  set -x
else
  export CCACHE_REMOTE_STORAGE="${CCACHE_REMOTE_STORAGE}|read-only"
fi
