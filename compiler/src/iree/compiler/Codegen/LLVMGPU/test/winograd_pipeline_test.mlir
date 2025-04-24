// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @winograd_filter_transform() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
  %3 = tensor.empty() : tensor<8x8x64x128xf32>
  %4 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%2 : tensor<3x3x64x128xf32>) outs(%3 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [8, 8, 64, 128], strides = [1, 1, 1, 1] : tensor<8x8x64x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
  return
}
//   CHECK-LABEL:  func.func @winograd_filter_transform
//     CHECK-NOT:      memref.alloc
//         CHECK:      vector.transfer_read
//         CHECK:      vector.transfer_read
//         CHECK:      scf.for
//         CHECK:        scf.for
//         CHECK:          vector.transfer_read
//         CHECK:          vector.contract
//         CHECK:          vector.contract
//         CHECK:          vector.transfer_write

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @winograd_input_transform() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
  %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
  %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x2x6x6x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
  return
}
//   CHECK-LABEL:  func.func @winograd_input_transform
//     CHECK-NOT:      memref.alloc
//         CHECK:      vector.transfer_read
//         CHECK:      vector.transfer_read
//         CHECK:      scf.for
//         CHECK:        scf.for
//         CHECK:          scf.for
//         CHECK:          vector.transfer_read
//         CHECK:          vector.contract
//         CHECK:          vector.contract
//         CHECK:          vector.transfer_write

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @winograd_output_transform() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>> -> tensor<8x8x2x6x6x128xf16>
  %3 = tensor.empty() : tensor<2x36x36x128xf16>
  %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 36, 36, 128], strides = [1, 1, 1, 1] : tensor<2x36x36x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
  return
}
//   CHECK-LABEL:  func.func @winograd_output_transform
//     CHECK-NOT:      memref.alloc
//         CHECK:      vector.transfer_read
//         CHECK:      vector.transfer_read
//         CHECK:      scf.for
//         CHECK:        scf.for
//         CHECK:          scf.for
//         CHECK:          vector.transfer_read
//         CHECK:          vector.contract
//         CHECK:          vector.contract
//         CHECK:          vector.transfer_write
