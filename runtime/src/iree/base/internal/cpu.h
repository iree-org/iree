// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_CPU_H_
#define IREE_BASE_INTERNAL_CPU_H_

#include <stddef.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Processor data query
//===----------------------------------------------------------------------===//

// Initializes cached CPU data using |temp_allocator| for any temporary
// allocations required during initialization.
void iree_cpu_initialize(iree_allocator_t temp_allocator);

// Initializes cached CPU data with the given fields.
// Extraneous fields will be ignored and unspecified fields will be set to zero.
void iree_cpu_initialize_with_data(iree_host_size_t field_count,
                                   const uint64_t* fields);

// Returns all fields up to IREE_CPU_DATA_FIELD_COUNT.
// Data will be zeroed until initialized with iree_cpu_initialize.
// See iree/schemas/cpu_data.h for interpretation.
const uint64_t* iree_cpu_data_fields(void);

// Returns the CPU data field or zero if the field is not available.
// Data will be zeroed until initialized with iree_cpu_initialize.
// See iree/schemas/cpu_data.h for interpretation.
//
// Common usage:
// if (iree_all_bits_set(iree_cpu_data_field(0),
//                       IREE_CPU_DATA0_ARM_64_DOTPROD)) {
//   // Bit is set and known true.
// } else {
//   // Bit is unset but _may_ be true (CPU data not available, etc).
// }
uint64_t iree_cpu_data_field(iree_host_size_t field);

// Queries the CPU data from the system and stores it into |out_fields|.
// Up to |field_count| fields will be written and if greater than the number
// available the remaining fields will be set to zero.
// See iree/schemas/cpu_data.h for interpretation.
void iree_cpu_read_data(iree_host_size_t field_count, uint64_t* out_fields);

// Looks up a canonical value in the CPU data fields.
// Keys are defined in iree/schemas/cpu_data.h.
iree_status_t iree_cpu_lookup_data_by_key(iree_string_view_t key,
                                          int64_t* IREE_RESTRICT out_value);

//===----------------------------------------------------------------------===//
// Processor identification
//===----------------------------------------------------------------------===//

typedef uint32_t iree_cpu_processor_id_t;
typedef uint32_t iree_cpu_processor_tag_t;

// Returns the ID of the logical processor executing this code.
iree_cpu_processor_id_t iree_cpu_query_processor_id(void);

// Returns the ID of the logical processor executing this code, using |tag| to
// memoize the query in cases where it does not change frequently.
// |tag| must be initialized to 0 on first call and may be reset to 0 by the
// caller at any time to invalidate the cached result.
void iree_cpu_requery_processor_id(iree_cpu_processor_tag_t* IREE_RESTRICT tag,
                                   iree_cpu_processor_id_t* IREE_RESTRICT
                                       processor_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_ARENA_H_
