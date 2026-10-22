// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Aliases for the FastRPC types used by the generated hexagon_dsp.h, based only
// on built-in C types. This decouples headers of the driver from the generated
// interface header. rpc_types.c statically checks that the aliases match the
// generated types.

#ifndef IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_
#define IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_

// Handle of a FastRPC session with the DSP (remote_handle64).
typedef unsigned long rpc_session_handle_t;

#endif  // IREE_HAL_DRIVERS_HEXAGON_RPC_TYPES_H_
