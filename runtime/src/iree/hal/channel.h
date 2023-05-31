// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CHANNEL_H_
#define IREE_HAL_CHANNEL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_channel_t
//===----------------------------------------------------------------------===//

enum iree_hal_channel_flag_bits_t {
  IREE_HAL_CHANNEL_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_channel_flags_t;

// Specifies that the channel should use environment settings if available.
#define IREE_HAL_CHANNEL_RANK_DEFAULT ((int32_t)-1)
#define IREE_HAL_CHANNEL_COUNT_DEFAULT ((int32_t)-1)

// Indicates that the rank will not be part of any group.
#define IREE_HAL_CHANNEL_NO_COLOR ((int32_t)-1)

// Parameters defining how a channel should be configured.
typedef struct {
  // Flags controlling channel behavior.
  iree_hal_channel_flags_t flags;
  // Implementation-defined identifier for the channel.
  // May be empty to indicate that the environment should be used to populate
  // the identifier.
  //
  // Equivalent to:
  //   ncclUniqueId
  iree_const_byte_span_t id;
  // User-defined group key for differentiating multiple channel groups.
  // Can be treated as opaque.
  iree_string_view_t group;
  // Rank of the participant within the collective group.
  // May be IREE_HAL_CHANNEL_RANK_DEFAULT to indicate that the environment
  // should be used to populate the rank.
  int32_t rank;
  // Total number of participants within the collective group.
  // May be IREE_HAL_CHANNEL_COUNT_DEFAULT to indicate that the environment
  // should be used to populate the count.
  int32_t count;
} iree_hal_channel_params_t;

// A collective communication channel representing a single rank.
//
// Equivalent to:
//   MPI_Comm
//   ncclComm_t
//   ccl::communicator
typedef struct iree_hal_channel_t iree_hal_channel_t;

// Creates a channel on |device| for use by all queues defined in
// |queue_affinity|. |params| may specify the channel parameters or leave its
// fields as default to indicate that the value should be sourced from the
// environment.
IREE_API_EXPORT iree_status_t iree_hal_channel_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

// Retains the given |channel| for the caller.
IREE_API_EXPORT void iree_hal_channel_retain(iree_hal_channel_t* channel);

// Releases the given |channel| from the caller.
IREE_API_EXPORT void iree_hal_channel_release(iree_hal_channel_t* channel);

// Splits |base_channel| into a subgroup based on |color| and |key|.
// Returns a NULL channel if color is IREE_HAL_CHANNEL_NO_COLOR indicating that
// the rank is not a participant in any subgroup.
//
// Equivalent to:
//   MPI_Comm_split
//   ncclCommSplit
IREE_API_EXPORT iree_status_t iree_hal_channel_split(
    iree_hal_channel_t* base_channel, int32_t color, int32_t key,
    iree_hal_channel_flags_t flags, iree_hal_channel_t** out_split_channel);

// Returns the rank the channel represents as a participant in a collective
// group in `[0, count)` and the total participant count.
IREE_API_EXPORT void iree_hal_channel_query_rank_and_count(
    const iree_hal_channel_t* channel, int32_t* out_rank, int32_t* out_count);

// Returns the rank the channel represents as a participant in a collective
// group in `[0, count)`.
IREE_API_EXPORT int32_t
iree_hal_channel_rank(const iree_hal_channel_t* channel);

// Returns the total participant count in a collective group.
IREE_API_EXPORT int32_t
iree_hal_channel_count(const iree_hal_channel_t* channel);

//===----------------------------------------------------------------------===//
// iree_hal_channel_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_channel_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_channel_t* channel);

  iree_status_t(IREE_API_PTR* split)(iree_hal_channel_t* base_channel,
                                     int32_t color, int32_t key,
                                     iree_hal_channel_flags_t flags,
                                     iree_hal_channel_t** out_split_channel);

  void(IREE_API_PTR* query_rank_and_count)(const iree_hal_channel_t* channel,
                                           int32_t* out_rank,
                                           int32_t* out_count);
} iree_hal_channel_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_channel_vtable_t);

IREE_API_EXPORT void iree_hal_channel_destroy(iree_hal_channel_t* channel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CHANNEL_H_
