// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-transform-dialect-interpreter)' \
// RUN:   --iree-gpu-test-target=sm_60 --iree-codegen-transform-dialect-library=%p/attention_transform_spec.mlir| \
// RUN: FileCheck --check-prefix=CHECK %s

module {
  func.func @_attention_dispatch_0() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.250000e-01 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
    %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
    %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
    %7 = tensor.empty() : tensor<192x1024x64xf16>
    %8 = iree_linalg_ext.attention ins(%4, %5, %6, %cst : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, f16) outs(%7 : tensor<192x1024x64xf16>) -> tensor<192x1024x64xf16>
    flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : tensor<192x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
    return
  }
}

// We can trust our decomposition for attention so we don't need to check all
// the IR. What we want to check is if there are any spurious allocations and
// use this test as a smoke test.

// CHECK:      func.func @_attention_dispatch_0()

// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index

// CHECK:        %[[Q:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[K:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[V:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[O:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// Read from global memory for Q
// CHECK:        %[[Q_REG:.+]] = vector.transfer_read %[[Q]]
// Loop
// CHECK:        scf.for %{{.*}} = %[[C0]] to %[[C1024]] step %[[C64]]
// Copy for K and V
// CHECK:         linalg.generic
// CHECK:           linalg.yield
// CHECK:         linalg.generic
// CHECK:           linalg.yield
// CHECK:         gpu.barrier
// Read from shared memory for K
// CHECK:         vector.transfer_read
// Matmul 1
// CHECK:         vector.contract
// Read from shared memory for V
// CHECK:         vector.transfer_read
// Matmul 2
// CHECK:         vector.contract
// CHECK:         scf.yield
// Write to output
// CHECK:       vector.transfer_write %{{.*}}, %[[O]]
