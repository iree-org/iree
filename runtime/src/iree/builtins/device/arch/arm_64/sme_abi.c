// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../device.h"

// This file contains stubs of the AArch64 SME ABI support routines, defined in
// the AAPCS64.
// See:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#81sme-support-routines.
//
// Note that use use of these stubs is safe for the code IREE currently
// generates, as each dispatch region is self-contained. SME dispatches do not
// call out to other SME-enabled functions, so ZA-state does not need to be
// preserved.
//
// The LLVM backend still plants calls to these functions, so these definitions
// are simply to satisfy the linker, but these functions won't ever be called.

typedef struct sme_state {
  int64_t x0;
  int64_t x1;
} sme_state_t;

typedef struct TPIDR2_block TPIDR2_block_t;

IREE_DEVICE_EXPORT sme_state_t __arm_sme_state(void) {
  // No-op as not needed yet.
  sme_state_t null_state = {0, 0};
  return null_state;
}

IREE_DEVICE_EXPORT void __arm_tpidr2_restore(TPIDR2_block_t* blk) {
  // No-op as not needed yet.
}

IREE_DEVICE_EXPORT void __arm_tpidr2_save(void) {
  // No-op as not needed yet.
}

IREE_DEVICE_EXPORT void __arm_za_disable(void) {
  // No-op as not needed yet.
}

// Unlike the stubs above, this one must return a real value: it's called
// from the prologue, *before* `smstart`, to save the pre-streaming VG for
// unwind info on targets lacking classic SVE, where `cntd` is illegal outside
// Streaming SVE mode. A normal host link gets this for free from compiler-rt's
// sme-abi.S. IREE's embedded linking is freestanding and has none, hence the
// local copy. See https://github.com/llvm/llvm-project/issues/204853.
IREE_DEVICE_EXPORT int64_t __arm_get_current_vg(void) {
  int64_t svcr;
  __asm__("mrs %0, SVCR" : "=r"(svcr));
  if (!(svcr & 1)) {
    // Not in Streaming SVE mode: no legal way to read VG here.
    return 0;
  }
  int64_t vg;
  __asm__("cntd %0" : "=r"(vg));
  return vg;
}
