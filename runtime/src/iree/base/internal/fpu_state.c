// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/fpu_state.h"

#include <stdbool.h>

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
#include <xmmintrin.h>
#endif  // IREE_ARCH_X86_*

#if defined(IREE_COMPILER_MSVC)
#include <intrin.h>
#endif  // IREE_COMPILER_MSVC

//==============================================================================
// iree_fpu_state_t
//==============================================================================
// https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/core/platform/denormal.cc
// https://chromium.googlesource.com/chromium/blink/+/main/Source/platform/audio/DenormalDisabler.h

static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero);

#if defined(IREE_ARCH_ARM_32)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x1000000) | (denormals_to_zero ? 0x1000000 : 0);
}
#elif defined(IREE_ARCH_ARM_64)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x1080000) | (denormals_to_zero ? 0x1080000 : 0);
}
#elif defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x8040) | (denormals_to_zero ? 0x8040 : 0);
}
#else
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return state;
}
#endif  // IREE_ARCH_*

static uint64_t iree_fpu_load_state(void);
static void iree_fpu_store_state(uint64_t state);

#if defined(IREE_ARCH_ARM_32) && defined(IREE_COMPILER_MSVC)
static uint64_t iree_fpu_load_state(void) {
  return (uint64_t)_MoveFromCoprocessor(10, 7, 1, 0, 0);
}
static void iree_fpu_store_state(uint64_t state) {
  _MoveToCoprocessor((int)state, 10, 7, 1, 0, 0);
}
#elif defined(IREE_ARCH_ARM_32)
static uint64_t iree_fpu_load_state() {
  uint32_t fpscr;
  __asm__ __volatile__("VMRS %[fpscr], fpscr" : [fpscr] "=r"(fpscr));
  return (uint64_t)fpscr;
}
static void iree_fpu_store_state(uint64_t state) {
  __asm__ __volatile__("VMSR fpscr, %[fpscr]" : : [fpscr] "r"(state));
}
#elif defined(IREE_ARCH_ARM_64) && defined(IREE_COMPILER_MSVC)
static uint64_t iree_fpu_load_state(void) {
  return (uint64_t)_ReadStatusReg(0x5A20);
}
static void iree_fpu_store_state(uint64_t state) {
  _WriteStatusReg(0x5A20, (__int64)state);
}
#elif defined(IREE_ARCH_ARM_64)
static uint64_t iree_fpu_load_state(void) {
  uint64_t fpcr;
  __asm__ __volatile__("MRS %[fpcr], fpcr" : [fpcr] "=r"(fpcr));
  return fpcr;
}
static void iree_fpu_store_state(uint64_t state) {
  __asm__ __volatile__("MSR fpcr, %[fpcr]" : : [fpcr] "r"(state));
}
#elif defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
static uint64_t iree_fpu_load_state(void) { return (uint64_t)_mm_getcsr(); }
static void iree_fpu_store_state(uint64_t state) {
  _mm_setcsr((unsigned int)state);
}
#else
static uint64_t iree_fpu_load_state(void) { return 0; }
static void iree_fpu_store_state(uint64_t state) {}
#endif  // IREE_ARCH_*

iree_fpu_state_t iree_fpu_state_push(iree_fpu_state_flags_t flags) {
  iree_fpu_state_t state;
  state.current_value = state.previous_value = iree_fpu_load_state();
  state.current_value = iree_fpu_state_set_dtz(
      state.current_value,
      (flags & IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO) ? true : false);
  if (state.previous_value != state.current_value) {
    iree_fpu_store_state(state.current_value);
  }
  return state;
}

void iree_fpu_state_pop(iree_fpu_state_t state) {
  if (state.previous_value != state.current_value) {
    iree_fpu_store_state(state.previous_value);
  }
}
