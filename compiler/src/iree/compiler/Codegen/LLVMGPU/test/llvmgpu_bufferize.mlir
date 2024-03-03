// RUN: iree-opt --iree-codegen-llvmgpu-bufferize --split-input-file %s | FileCheck %s

module {
  func.func @bufferize_thread_local(%arg0: index) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_ved = arith.constant dense<0.000000e+00> : vector<1x1x4x4xf16>
    %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
    %2 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg0, %arg0, %arg0], sizes = [1, 1, 8, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>> -> tensor<1x1x8x64xf16>
    %3 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<1xf16>
    %4 = scf.forall (%arg1, %arg2) in (2, 16) shared_outs(%arg3 = %2) -> (tensor<1x1x8x64xf16>) {
      %5 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
      %extracted_slice = tensor.extract_slice %arg3[0, 0, %5, %6] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x1x8x64xf16> to tensor<1x1x4x4xf16>
      %alloc_tensor = bufferization.alloc_tensor() : tensor<1x1x4x4xf16>
      %copy = bufferization.materialize_in_destination %extracted_slice in %alloc_tensor : (tensor<1x1x4x4xf16>, tensor<1x1x4x4xf16>) -> tensor<1x1x4x4xf16>
      %7 = vector.transfer_read %3[%c0], %cst {in_bounds = [true]} : tensor<1xf16>, vector<1xf16>
      %8 = vector.broadcast %7 : vector<1xf16> to vector<1x1x4x4xf16>
      %9 = vector.transfer_read %arg3[%c0, %c0, %5, %6], %cst {in_bounds = [true, true, true, true]} : tensor<1x1x8x64xf16>, vector<1x1x4x4xf16>
      %10 = arith.addf %9, %8 : vector<1x1x4x4xf16>
      %11 = vector.transfer_write %10, %copy[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf16>, tensor<1x1x4x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %11 into %arg3[0, 0, %5, %6] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x1x4x4xf16> into tensor<1x1x8x64xf16>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    flow.dispatch.tensor.store %4, %1, offsets = [%arg0, %arg0, %arg0, %arg0], sizes = [1, 1, 8, 64], strides = [1, 1, 1, 1] : tensor<1x1x8x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
    return
  }
}
// CHECK-LABEL: func.func @bufferize_thread_local
//       CHECK:   scf.forall {{.*}} in (2, 16) {
//       CHECK:     %[[ALLOC:.+]] = memref.alloc() : memref<1x1x4x4xf16, #gpu.address_space<private>>
//       CHECK:     memref.copy %{{.*}}, %[[ALLOC]]
//  CHECK-SAME:       memref<1x1x4x4xf16, strided<[1310720, 4096, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
//  CHECK-SAME:       to memref<1x1x4x4xf16, #gpu.address_space<private>>
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
