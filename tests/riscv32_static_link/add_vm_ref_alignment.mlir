// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-compile %s \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv32-unknown-elf \
// RUN:   --iree-llvmcpu-target-abi=ilp32 \
// RUN:   --iree-llvmcpu-target-cpu-features=+m,+f \
// RUN:   --iree-vm-target-index-bits=32 \
// RUN:   --iree-llvmcpu-debug-symbols=false \
// RUN:   --iree-vm-bytecode-module-strip-source-map=true \
// RUN:   --iree-llvmcpu-link-embedded=false \
// RUN:   --output-format=vm-c \
// RUN:   --iree-vm-c-module-output-format=mlir-text \
// RUN:   -o - | FileCheck %s

// Regression test for RV32 VM-C static import ABI packing. The iICrD
// hal.fence.await shim must align iree_vm_ref_t to 4 bytes on RV32.

// CHECK-LABEL: emitc.func private @module_call_0iICrD_i_1_import_shim(
// CHECK-SAME: %[[COUNT_ARG:arg[0-9]+]]: i32, %[[REF_ARG:arg[0-9]+]]: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
// CHECK:      assign %[[COUNT_ARG]] : i32 to %{{[0-9]+}} : <i32>
// CHECK:      %[[REF_PTR_AS_UINT:[0-9]+]] = cast %{{[0-9]+}} : !emitc.ptr<ui8> to !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[REF_PTR_ALIGN:[0-9]+]] = literal "4" : !emitc.opaque<"iree_host_size_t">
// CHECK-NEXT: %[[REF_PTR_ALIGNED:[0-9]+]] = call_opaque "iree_host_align"(%[[REF_PTR_AS_UINT]], %[[REF_PTR_ALIGN]]) : (!emitc.opaque<"uintptr_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[REF_BYTES:[0-9]+]] = cast %[[REF_PTR_ALIGNED]] : !emitc.opaque<"uintptr_t"> to !emitc.ptr<ui8>
// CHECK-NEXT: %[[REF_SLOT:[0-9]+]] = cast %[[REF_BYTES]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
// CHECK-NEXT: call_opaque "iree_vm_ref_retain"(%[[REF_ARG]], %[[REF_SLOT]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

module {
  func.func @test(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
    %result = arith.addf %lhs, %rhs : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
