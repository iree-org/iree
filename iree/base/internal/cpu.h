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
// iree_cpu_*
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
