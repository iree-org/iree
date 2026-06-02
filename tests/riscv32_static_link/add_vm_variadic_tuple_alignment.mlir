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

// Regression test for RV32 VM-C static import ABI packing. The rIrrrICrIID
// hal.device.queue.execute.indirect shim must preserve the alignment boundary
// for each variadic rII tuple element.

// CHECK-LABEL: emitc.func private @module_call_0rIrrrICrIID_v_3_import_shim(
// CHECK-SAME: %[[COUNT_ARG:arg[0-9]+]]: i32, %[[REF_ARG:arg[0-9]+]]: !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
// CHECK:      assign %[[COUNT_ARG]] : i32 to %{{[0-9]+}} : <i32>
// CHECK:      %[[AFTER_COUNT:[0-9]+]] = add %{{[0-9]+}}, %{{[0-9]+}} : (!emitc.ptr<ui8>, !emitc.opaque<"iree_host_size_t">) -> !emitc.ptr<ui8>
// CHECK-NEXT: %[[TUPLE_PTR:[0-9]+]] = cast %[[AFTER_COUNT]] : !emitc.ptr<ui8> to !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[TUPLE_ALIGN:[0-9]+]] = literal "8" : !emitc.opaque<"iree_host_size_t">
// CHECK-NEXT: %[[TUPLE_ALIGNED:[0-9]+]] = call_opaque "iree_host_align"(%[[TUPLE_PTR]], %[[TUPLE_ALIGN]]) : (!emitc.opaque<"uintptr_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[TUPLE_BYTES:[0-9]+]] = cast %[[TUPLE_ALIGNED]] : !emitc.opaque<"uintptr_t"> to !emitc.ptr<ui8>
// CHECK-NEXT: %[[REF_PTR:[0-9]+]] = cast %[[TUPLE_BYTES]] : !emitc.ptr<ui8> to !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[REF_ALIGN:[0-9]+]] = literal "4" : !emitc.opaque<"iree_host_size_t">
// CHECK-NEXT: %[[REF_ALIGNED:[0-9]+]] = call_opaque "iree_host_align"(%[[REF_PTR]], %[[REF_ALIGN]]) : (!emitc.opaque<"uintptr_t">, !emitc.opaque<"iree_host_size_t">) -> !emitc.opaque<"uintptr_t">
// CHECK-NEXT: %[[REF_BYTES:[0-9]+]] = cast %[[REF_ALIGNED]] : !emitc.opaque<"uintptr_t"> to !emitc.ptr<ui8>
// CHECK-NEXT: %[[REF_SLOT:[0-9]+]] = cast %[[REF_BYTES]] : !emitc.ptr<ui8> to !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
// CHECK-NEXT: call_opaque "iree_vm_ref_retain"(%[[REF_ARG]], %[[REF_SLOT]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()

module {
  func.func @test(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
    %result = arith.addf %lhs, %rhs : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
