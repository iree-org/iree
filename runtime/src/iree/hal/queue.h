// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_QUEUE_H_
#define IREE_HAL_QUEUE_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitmap indicating logical device queue affinity.
// Used to direct submissions to specific device queues or locate memory nearby
// where it will be used. The meaning of the bits in the bitmap is
// implementation-specific: a bit may represent a logical queue in an underlying
// API such as a VkQueue or a physical queue such as a discrete virtual device.
//
// Bitwise operations can be performed on affinities; for example AND'ing two
// affinities will produce the intersection and OR'ing will produce the union.
// This enables just-in-time selection as a command buffer could be made
// available to some set of queues when recorded and then AND'ed with an actual
// set of queues to execute on during submission.
typedef uint64_t iree_hal_queue_affinity_t;

// Specifies that any queue may be selected.
#define IREE_HAL_QUEUE_AFFINITY_ANY ((iree_hal_queue_affinity_t)(-1))

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_QUEUE_H_
