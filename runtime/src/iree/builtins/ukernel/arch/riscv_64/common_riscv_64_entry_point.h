// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_ENTRY_POINT_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_ENTRY_POINT_H_

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
#define IREE_UK_BUILD_RISCV_64_RVV
#else
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h"
#endif

#endif // IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_ENTRY_POINT_H_
