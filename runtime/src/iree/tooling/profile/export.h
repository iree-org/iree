// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_EXPORT_H_
#define IREE_TOOLING_PROFILE_EXPORT_H_

#include "iree/tooling/profile/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reads a profile bundle from |path| and writes a schema-versioned decoded
// interchange export to |output_path|. Unlike command-local JSONL reports,
// export rows use a stable |record_type| namespace for downstream tooling.
iree_status_t iree_profile_export_file(iree_string_view_t path,
                                       iree_string_view_t format,
                                       iree_string_view_t output_path,
                                       iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_EXPORT_H_
