// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_IREE_PROFILE_ATT_H_
#define IREE_TOOLS_IREE_PROFILE_ATT_H_

#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Decodes AMDGPU ATT/SQTT records in |path| and prints an annotated report.
iree_status_t iree_profile_att_file(iree_string_view_t path,
                                    iree_string_view_t format,
                                    iree_string_view_t filter, int64_t id,
                                    iree_string_view_t rocm_library_path,
                                    FILE* file,
                                    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_IREE_PROFILE_ATT_H_
