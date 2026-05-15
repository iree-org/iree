// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-llvmgpu-bufferization-pipeline))" --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @bufferize_with_thread_private_memory(%arg0: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [%arg0, %arg0, %arg0, %arg0], sizes = [1, 1, 8, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>> -> tensor<1x1x8x64xf16>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%arg0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<1xf16>
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
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [%arg0, %arg0, %arg0, %arg0], sizes = [1, 1, 8, 64], strides = [1, 1, 1, 1] : tensor<1x1x8x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
  return
}
// CHECK-LABEL: func.func @bufferize_with_thread_private_memory
//       CHECK:   scf.forall {{.*}} in (2, 16) {
//       CHECK:     %[[ALLOC:.+]] = memref.alloca() : memref<1x1x4x4xf16, #gpu.address_space<private>>
//       CHECK:     memref.copy %{{.*}}, %[[ALLOC]]
//  CHECK-SAME:       memref<1x1x4x4xf16, strided<[1310720, 4096, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
//  CHECK-SAME:       to memref<1x1x4x4xf16, #gpu.address_space<private>>
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}

// -----

#pipeline_layout_scatter = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @linalg_ext_scatter_preserves_original() {
  %c0 = arith.constant 0 : index
  %updates_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>>
  %indices_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>>
  %original_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>>
  %result_dst = hal.interface.binding.subspan layout(#pipeline_layout_scatter) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  %updates = iree_tensor_ext.dispatch.tensor.load %updates_src, offsets = [0, 0, 0], sizes = [11, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>> -> tensor<11x1x512xf32>
  %indices = iree_tensor_ext.dispatch.tensor.load %indices_src, offsets = [0], sizes = [11], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>> -> tensor<11xi32>
  %original = iree_tensor_ext.dispatch.tensor.load %original_src, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>> -> tensor<32x1x512xf32>
  %scatter = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%updates, %indices : tensor<11x1x512xf32>, tensor<11xi32>) outs(%original : tensor<32x1x512xf32>) {
  ^bb0(%update: f32, %original_value: f32):
    iree_linalg_ext.yield %update : f32
  } -> tensor<32x1x512xf32>
  iree_tensor_ext.dispatch.tensor.store %scatter, %result_dst, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : tensor<32x1x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  return
}
// CHECK-LABEL: func.func @linalg_ext_scatter_preserves_original
// CHECK:       %[[ORIGINAL_SUBSPAN:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK:       %[[ORIGINAL:.+]] = memref.assume_alignment %[[ORIGINAL_SUBSPAN]]
// CHECK:       %[[RESULT:.+]] = hal.interface.binding.subspan {{.*}} binding(3)
// CHECK:       %[[ALLOC:.+]] = memref.alloc()
// CHECK:       memref.copy %[[ORIGINAL]], %[[ALLOC]]
// CHECK:       iree_linalg_ext.scatter
// CHECK-SAME:    outs(%[[ALLOC]]

// -----

#pipeline_layout_scatter_combine = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @linalg_ext_scatter_combiner_reads_original() {
  %c0 = arith.constant 0 : index
  %updates_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_combine) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>>
  %indices_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_combine) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>>
  %original_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_combine) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>>
  %result_dst = hal.interface.binding.subspan layout(#pipeline_layout_scatter_combine) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  %updates = iree_tensor_ext.dispatch.tensor.load %updates_src, offsets = [0, 0, 0], sizes = [11, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>> -> tensor<11x1x512xf32>
  %indices = iree_tensor_ext.dispatch.tensor.load %indices_src, offsets = [0], sizes = [11], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>> -> tensor<11xi32>
  %original = iree_tensor_ext.dispatch.tensor.load %original_src, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>> -> tensor<32x1x512xf32>
  %scatter = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%updates, %indices : tensor<11x1x512xf32>, tensor<11xi32>) outs(%original : tensor<32x1x512xf32>) {
  ^bb0(%update: f32, %original_value: f32):
    %combined = arith.addf %update, %original_value : f32
    iree_linalg_ext.yield %combined : f32
  } -> tensor<32x1x512xf32>
  iree_tensor_ext.dispatch.tensor.store %scatter, %result_dst, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : tensor<32x1x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  return
}
// CHECK-LABEL: func.func @linalg_ext_scatter_combiner_reads_original
// CHECK:       %[[ORIGINAL_SUBSPAN:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK:       %[[ORIGINAL:.+]] = memref.assume_alignment %[[ORIGINAL_SUBSPAN]]
// CHECK:       %[[ALLOC:.+]] = memref.alloc()
// CHECK:       memref.copy %[[ORIGINAL]], %[[ALLOC]]
// CHECK:       iree_linalg_ext.scatter
// CHECK-SAME:    outs(%[[ALLOC]]

// -----

#pipeline_layout_scatter_mask = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @linalg_ext_scatter_masked_update_preserves_original() {
  %c0 = arith.constant 0 : index
  %updates_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_mask) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>>
  %indices_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_mask) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>>
  %mask_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_mask) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi1>>
  %original_src = hal.interface.binding.subspan layout(#pipeline_layout_scatter_mask) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>>
  %result_dst = hal.interface.binding.subspan layout(#pipeline_layout_scatter_mask) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  %updates = iree_tensor_ext.dispatch.tensor.load %updates_src, offsets = [0, 0, 0], sizes = [11, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11x1x512xf32>> -> tensor<11x1x512xf32>
  %indices = iree_tensor_ext.dispatch.tensor.load %indices_src, offsets = [0], sizes = [11], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi32>> -> tensor<11xi32>
  %mask = iree_tensor_ext.dispatch.tensor.load %mask_src, offsets = [0], sizes = [11], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<11xi1>> -> tensor<11xi1>
  %original = iree_tensor_ext.dispatch.tensor.load %original_src, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x512xf32>> -> tensor<32x1x512xf32>
  %scatter = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%updates, %indices, %mask : tensor<11x1x512xf32>, tensor<11xi32>, tensor<11xi1>) outs(%original : tensor<32x1x512xf32>) {
  ^bb0(%update: f32, %original_value: f32):
    iree_linalg_ext.yield %update : f32
  } -> tensor<32x1x512xf32>
  iree_tensor_ext.dispatch.tensor.store %scatter, %result_dst, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : tensor<32x1x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x512xf32>>
  return
}
// CHECK-LABEL: func.func @linalg_ext_scatter_masked_update_preserves_original
// CHECK:       %[[ORIGINAL_SUBSPAN:.+]] = hal.interface.binding.subspan {{.*}} binding(3)
// CHECK:       %[[ORIGINAL:.+]] = memref.assume_alignment %[[ORIGINAL_SUBSPAN]]
// CHECK:       %[[ALLOC:.+]] = memref.alloc()
// CHECK:       memref.copy %[[ORIGINAL]], %[[ALLOC]]
// CHECK:       iree_linalg_ext.scatter
// CHECK-SAME:    outs(%[[ALLOC]]
