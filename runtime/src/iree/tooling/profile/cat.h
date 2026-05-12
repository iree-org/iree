// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_CAT_H_
#define IREE_TOOLING_PROFILE_CAT_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reads a profile bundle from |path| and writes raw record diagnostics.
iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                    iree_string_view_t format, FILE* file,
                                    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_CAT_H_
