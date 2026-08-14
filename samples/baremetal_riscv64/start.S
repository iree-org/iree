// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal boot shim for QEMU's virt machine.
// QEMU's generic loader sets the PC to the ELF entry point but does not
// initialize sp. newlib's crt0 (_start) expects sp to be valid. Enable the
// FPU as well: mstatus.FS starts Off, so FP instructions would otherwise trap.
    .section .text.enter, "ax", @progbits
    .globl _enter
_enter:
    li t0, 0x6000        // mstatus.FS = 11 (Dirty): enable FPU
    csrs mstatus, t0
    la sp, __stack_top
    tail _start
