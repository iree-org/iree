// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Re-export backtrace.h from BCR's libbacktrace.
// This wrapper exists to properly export the header for Bazel's layering check.
// When status_stack_trace.c includes <backtrace.h>, the layering check needs
// to see that the include comes from a direct dependency's exported header.
// BCR's libbacktrace exports backtrace.h, but cc_library deps don't
// transitively propagate include paths. This wrapper header bridges the gap.

#ifndef IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_
#define IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_

#include <backtrace.h>

#endif  // IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_
