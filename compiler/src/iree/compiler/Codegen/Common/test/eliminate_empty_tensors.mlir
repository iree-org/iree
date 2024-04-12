// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors))" %s | FileCheck %s

// -----
func.func @eliminate_empty_tensors_with_store_op() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  %1 = tensor.empty() : tensor<32x384xf32>
  scf.for %arg0 = %c0 to %c128 step %c32 {
    %2 = scf.for %arg1 = %c0 to %c32 step %c8 iter_args(%arg2 = %1) -> (tensor<32x384xf32>) {
      scf.yield %arg2 : tensor<32x384xf32>
    }
    flow.dispatch.tensor.store %2, %0, offsets = [%arg0, 0], sizes = [32, 384], strides = [1, 1] : tensor<32x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  }
  return
}

// CHECK-LABEL: @eliminate_empty_tensors_with_store_op
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK: %[[C32:.+]] = arith.constant 32 : index
// CHECK: %[[C128:.+]] = arith.constant 128 : index
// CHECK: %[[SPAN:.+]] = hal.interface.binding.subspan
// CHECK: scf.for %[[ARG0:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SPAN]], offsets = [%[[ARG0]], 0]
// CHECK:   %[[RES:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C8]] iter_args(%{{.+}} = %[[LOAD]])
// CHECK:   flow.dispatch.tensor.store %[[RES]], %[[SPAN]]
