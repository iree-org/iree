// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Checks statically that the type aliases defined in rpc_types.h match the
// types in the hexagon_dsp.h header generated from the RPC interface.

#include "iree/hal/drivers/hexagon/rpc_types.h"

#include "hexagon_dsp.h"

#define IREE_HAL_HEXAGON_ASSERT_TYPES_EQ(a, b)       \
  _Static_assert(__builtin_types_compatible_p(a, b), \
                 "FastRPC type mismatch, fix type in rpc_types.h")

IREE_HAL_HEXAGON_ASSERT_TYPES_EQ(remote_handle64, rpc_session_handle_t);
