// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>
module {
  func.func @winograd_filter_transform() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x64x128xf32>> -> tensor<3x3x64x128xf32>
    %3 = tensor.empty() : tensor<8x8x64x128xf32>
    %4 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%2 : tensor<3x3x64x128xf32>) outs(%3 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [8, 8, 64, 128], strides = [1, 1, 1, 1] : tensor<8x8x64x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8x64x128xf32>>
    return
  }
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

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>
module {
  func.func @winograd_input_transform() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x34x34x128xf16>> -> tensor<2x34x34x128xf16>
    %3 = tensor.empty() : tensor<8x8x2x6x6x128xf16>
    %4 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<2x34x34x128xf16>) outs(%3 : tensor<8x8x2x6x6x128xf16>) -> tensor<8x8x2x6x6x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x2x6x6x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<8x8x2x6x6x128xf16>>
    return
  }
}
//   CHECK-LABEL:  func.func @winograd_input_transform
//     CHECK-NOT:      memref.alloc
//         CHECK:      scf.for
//         CHECK:        vector.transfer_read
//         CHECK:        vector.transfer_read
//         CHECK:        scf.for
//         CHECK:          scf.for
//         CHECK:          vector.transfer_read
//         CHECK:          vector.contract
//         CHECK:          vector.contract
//         CHECK:          vector.transfer_write

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100"}>
module {
  func.func @winograd_output_transform() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 2, 6, 6, 128], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<8x8x2x6x6x128xf16>> -> tensor<8x8x2x6x6x128xf16>
    %3 = tensor.empty() : tensor<2x36x36x128xf16>
    %4 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%2 : tensor<8x8x2x6x6x128xf16>) outs(%3 : tensor<2x36x36x128xf16>) -> tensor<2x36x36x128xf16>
    flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [2, 36, 36, 128], strides = [1, 1, 1, 1] : tensor<2x36x36x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x36x36x128xf16>>
    return
  }
}
//   CHECK-LABEL:  func.func @winograd_output_transform
//     CHECK-NOT:      memref.alloc
//         CHECK:      scf.for
//         CHECK:        vector.transfer_read
//         CHECK:        vector.transfer_read
//         CHECK:        scf.for
//         CHECK:          scf.for
//         CHECK:          vector.transfer_read
//         CHECK:          vector.contract
//         CHECK:          vector.contract
//         CHECK:          vector.transfer_write
