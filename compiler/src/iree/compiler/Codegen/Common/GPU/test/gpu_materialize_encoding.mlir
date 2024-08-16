// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-materialize-device-encoding))" --split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// 1. MFMA_F32_16x16x4_F32
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree.gpu.target = #iree_gpu.target<arch = "gfx942",
    features = "",
    wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
    storage =  b64|b32|b16|b8,
    subgroup =  shuffle|arithmetic,
    dot =  dp4xi8toi32,
    mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
    subgroup_size_choices = [64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [65535, 65535, 65535]>>
}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @set_encoding_LHS() attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_LHS
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<33x64x16x4xf32>
// CHECK:         %[[PACK:.*]] = tensor.pack %2 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 4] into %[[EMPTY]] : tensor<255x513xf32> -> tensor<33x64x16x4xf32>
// CHECK:         %[[EXPAND_LHS:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:      output_shape [33, 64, 16, 1, 4, 1] : tensor<33x64x16x4xf32> into tensor<33x64x16x1x4x1xf32>
// CHECK:         %[[EMPTY_LHS2:.*]] = tensor.empty() : tensor<33x64x4x16x1x1xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose ins(%[[EXPAND_LHS]] : tensor<33x64x16x1x4x1xf32>) outs(%[[EMPTY_LHS2]] : tensor<33x64x4x16x1x1xf32>) permutation = [0, 1, 4, 2, 5, 3]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK:         %[[EXPAND_LHS_2:.*]] = tensor.expand_shape %[[COLLAPSE]]
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_LHS_2]]

func.func @set_encoding_RHS() attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_RHS
// CHECK:         %[[EMPTY_RHS:.*]] = tensor.empty() : tensor<33x64x16x4xf32>
// CHECK:         %[[PACK_RHS:.*]] = tensor.pack %2 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 4] into %3 : tensor<255x513xf32> -> tensor<33x64x16x4xf32>
// CHECK:         %[[EXPAND_RHS:.*]] = tensor.expand_shape %[[PACK_RHS]]
// CHECK-SAME:      output_shape [33, 64, 16, 1, 4, 1] : tensor<33x64x16x4xf32> into tensor<33x64x16x1x4x1xf32>
// CHECK:         %[[EMPTY_RHS2:.*]] = tensor.empty() : tensor<33x64x4x16x1x1xf32>
// CHECK:         %[[TRANSPOSE_RHS:.*]] = linalg.transpose ins(%[[EXPAND_RHS]] : tensor<33x64x16x1x4x1xf32>) outs(%[[EMPTY_RHS2]] : tensor<33x64x4x16x1x1xf32>) permutation = [0, 1, 4, 2, 5, 3]
// CHECK:         %[[COLLAPSE_RHS:.*]] = tensor.collapse_shape %[[TRANSPOSE_RHS]]
// CHECK:         %[[EXPAND_RHS_2:.*]] = tensor.expand_shape %[[COLLAPSE_RHS]]
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_RHS_2]]

func.func @set_encoding_ACC() attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_ACC
// CHECK:         %[[EMPTY_ACC:.*]] = tensor.empty() : tensor<33x64x16x4xf32>
// CHECK:         %[[PACK_ACC:.*]] = tensor.pack %2 padding_value(%cst : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 4] into %[[EMPTY_ACC]] : tensor<255x513xf32> -> tensor<33x64x16x4xf32>
// CHECK:         %[[EXPAND_ACC:.*]] = tensor.expand_shape %[[PACK_ACC]]
// CHECK:         %[[EMPTY_ACC2:.*]] = tensor.empty() : tensor<33x64x4x16x1x1xf32>
// CHECK:         %[[TRANSPOSE_ACC:.*]] = linalg.transpose ins(%[[EXPAND_ACC]] : tensor<33x64x16x1x4x1xf32>) outs(%[[EMPTY_ACC2]] : tensor<33x64x4x16x1x1xf32>) permutation = [0, 1, 4, 2, 5, 3]
// CHECK:         %[[COLLAPSE_RHS:.*]] = tensor.collapse_shape %[[TRANSPOSE_ACC]]
// CHECK:         %[[EXPAND_ACC_2:.*]] = tensor.expand_shape %[[COLLAPSE_RHS]]
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_ACC_2]]