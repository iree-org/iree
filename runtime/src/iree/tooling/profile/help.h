// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_HELP_H_
#define IREE_TOOLING_PROFILE_HELP_H_

#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

const char* iree_profile_usage_text(void);
void iree_profile_fprint_usage(FILE* file);
void iree_profile_print_agent_markdown(FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_HELP_H_
