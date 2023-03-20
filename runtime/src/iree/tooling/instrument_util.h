// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_INSTRUMENT_UTIL_H_
#define IREE_TOOLING_INSTRUMENT_UTIL_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instrument data management
//===----------------------------------------------------------------------===//

// Processes instrument data in |context| based on command line flags.
// No-op if there's no instrument data available.
iree_status_t iree_tooling_process_instrument_data(
    iree_vm_context_t* context, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_INSTRUMENT_UTIL_H_
