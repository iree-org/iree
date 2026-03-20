// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Processor-level hints for spin loops.
//
// iree_processor_yield() emits a sub-nanosecond CPU hint that the calling
// thread is in a spin-wait loop. This reduces power consumption and avoids
// starving the sibling hyperthread on SMT cores. Much lighter than
// iree_thread_yield() (sched_yield/SwitchToThread) which involves a kernel
// context switch.
//
// Architecture coverage:
//   x86:    PAUSE (reduces speculative pipeline waste in spin loops)
//   ARM:    YIELD (hints the core to deprioritize this thread briefly)
//   RISC-V: PAUSE (Zihintpause extension, when available)
//   Other:  No-op (spins without hint)

#ifndef IREE_BASE_THREADING_PROCESSOR_H_
#define IREE_BASE_THREADING_PROCESSOR_H_

#include "iree/base/attributes.h"
#include "iree/base/target_platform.h"

#if defined(IREE_COMPILER_MSVC_COMPAT)
#include <intrin.h>
#endif  // IREE_COMPILER_MSVC_COMPAT

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_processor_yield(void) {
#if defined(IREE_COMPILER_MSVC_COMPAT)

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  _mm_pause();
#elif defined(IREE_ARCH_ARM_64)
  __yield();
#else
  // No intrinsic available; spins without hint.
#endif  // IREE_ARCH_*

#else  // GCC/Clang

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  asm volatile("pause");
#elif defined(IREE_ARCH_ARM_32) || defined(IREE_ARCH_ARM_64)
  asm volatile("yield");
#elif (defined(IREE_ARCH_RISCV_32) || defined(IREE_ARCH_RISCV_64)) && \
    defined(__riscv_zihintpause)
  asm volatile("pause");
#else
  // No instruction available; spins without hint.
#endif  // IREE_ARCH_*

#endif  // IREE_COMPILER_*
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_THREADING_PROCESSOR_H_
