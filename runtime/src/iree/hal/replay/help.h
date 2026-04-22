// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_HELP_H_
#define IREE_HAL_REPLAY_HELP_H_

#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns usage text for tools that capture `.ireereplay` files while running
// modules.
const char* iree_hal_replay_capture_usage_text(void);

// Returns usage text for `iree-run-replay`.
const char* iree_hal_replay_run_usage_text(void);

// Returns usage text for `iree-benchmark-replay`.
const char* iree_hal_replay_benchmark_usage_text(void);

// Returns usage text for `iree-dump-replay`.
const char* iree_hal_replay_dump_usage_text(void);

// Prints a Markdown playbook for humans and agents using HAL replay capture,
// execution, benchmarking, and dump tooling.
void iree_hal_replay_print_agent_markdown(FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_HELP_H_
