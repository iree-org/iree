// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_UTILS_CPU_FEATURES_H_
#define IREE_TOOLS_UTILS_CPU_FEATURES_H_

#include "iree/base/allocator.h"
#include "iree/base/status.h"
#include "iree/base/string_view.h"

// An opaque CPU-architecture-dependent context structure for CPU feature
// queries. Typically, each CPU architecture exposes certain CPUID registers,
// from which the CPU feature queries can be easily served by a few bit
// operations.
//
// What can be expensive is reading the CPUID registers themselves,
// particularly on architectures such as ARM where userspace code is not
// allowed to directly read them, so that some potentially expensive kernel
// mechanism has to kick in to actually expose that information to us.
//
// This struct would typically contain cached values of such CPUID registers.
typedef struct iree_cpu_features_t iree_cpu_features_t;

// On success, *features is a new iree_cpu_features_t.
// Must be destroyed by iree_cpu_features_free.
iree_status_t iree_cpu_features_allocate(iree_allocator_t allocator,
                                         iree_cpu_features_t** cpu_features);

// Destroys a iree_cpu_features_t that was created by
// iree_cpu_features_allocate.
void iree_cpu_features_free(iree_allocator_t allocator,
                            iree_cpu_features_t* cpu_features);

// On success, *result contains true if and only if the named feature is
// supported by the CPU. cpu_features must have been previously created by
// iree_cpu_features_allocate.
iree_status_t iree_cpu_features_query(iree_cpu_features_t* cpu_features,
                                      iree_string_view_t feature, bool* result);

#endif  // IREE_TOOLS_UTILS_CPU_FEATURES_H_
