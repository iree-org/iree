// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Public API: P2P operations (defined in iree_hal_streaming_device.c)
//===----------------------------------------------------------------------===//

// The P2P query functions are already defined in iree_hal_streaming_device.c:
// - cuDeviceCanAccessPeer
// - cuDeviceGetP2PAttribute

// The P2P context functions are already defined in
// iree_hal_streaming_context.c:
// - cuCtxEnablePeerAccess
// - cuCtxDisablePeerAccess

// The P2P memory copy function is already defined in
// iree_hal_streaming_memory.c:
// - cuMemcpyPeer
// - cuMemcpyPeerAsync

// This file is reserved for any additional P2P-specific internal
// implementation details that may be needed.
