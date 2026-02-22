// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Umbrella header for the iree/async/ API.
//
// Includes all core async types, the proactor interface, operation subtypes,
// and the frontier system. This is the single include for consumers that want
// the full async API surface.
//
// Utility headers (util/proactor_thread.h, util/operation_pool.h,
// util/sequence_emulation.h) are NOT included here — they are optional
// components with separate BUILD targets. Include them explicitly if needed.

#ifndef IREE_ASYNC_API_H_
#define IREE_ASYNC_API_H_

// Base types and platform primitives.
#include "iree/async/primitive.h"
#include "iree/async/types.h"

// Memory registration and scatter-gather.
#include "iree/async/region.h"
#include "iree/async/span.h"

// Resource handles.
#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/socket.h"

// Synchronization.
#include "iree/async/semaphore.h"

// Operations (base type and all subtypes).
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"

// Proactor (submission, completion, resource management).
#include "iree/async/proactor.h"

// Frontier system (vector clocks and causal ordering).
#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"

#endif  // IREE_ASYNC_API_H_
