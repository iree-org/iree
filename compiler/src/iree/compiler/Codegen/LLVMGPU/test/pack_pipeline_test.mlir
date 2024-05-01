// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target))" %s | FileCheck %s

#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {iree.gpu.target = #iree_gpu.alias_target<"sm_60">}>
module {
  func.func @static_pack() attributes {hal.executable.target = #executable_target_cuda_nvptx_fb} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<128x256xi32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4x16x16x32xi32>>
    %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xi32>> -> tensor<128x256xi32>
    %3 = tensor.empty() : tensor<4x16x16x32xi32>
    %pack = tensor.pack %2 inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %3 : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
    flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 16, 16, 32], strides = [1, 1, 1, 1] : tensor<4x16x16x32xi32> -> !flow.dispatch.tensor<writeonly:tensor<4x16x16x32xi32>>
    return
  }
}
//   CHECK-LABEL:  func.func @static_pack
//     CHECK-NOT:    vector.transfer_write
//     CHECK-NOT:    vector.transfer_read
//         CHECK:    scf.for
//         CHECK:      vector.transfer_read
//         CHECK:      vector.transpose
//         CHECK:      vector.transfer_write
