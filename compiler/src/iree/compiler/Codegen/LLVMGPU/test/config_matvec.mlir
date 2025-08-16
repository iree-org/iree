// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s --check-prefix=CDNA3


#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>
func.func @static_batch_matvec() {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = arith.index_castui %0 : i32 to index
  %4 = arith.index_castui %1 : i32 to index
  %5 = arith.index_castui %2 : i32 to index
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%5) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
  %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%3) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x1024xf16>>
  %8 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%4) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1024x128xf16>>
  %9 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [32, 1, 1024], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x1024xf16>> -> tensor<32x1x1024xf16>
  %10 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0, 0], sizes = [32, 1024, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1024x128xf16>> -> tensor<32x1024x128xf16>
  %11 = tensor.empty() : tensor<32x1x128xf16>
  %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  %13 = linalg.batch_matmul ins(%9, %10 : tensor<32x1x1024xf16>, tensor<32x1024x128xf16>) outs(%12 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  iree_tensor_ext.dispatch.tensor.store %13, %6, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
  return
}


// CHECK:     LLVMGPUWarpReduction
// CDNA3:     LLVMGPUTileAndFuse

// We want to deprecate LLVMGPUWarpReduction. Currently LLVMGPUVectorDistribution is not chosen in setReductionVectorDistributionConfig because it fails in 'hasReductionIterator' (which doesn't check specialized ops). This might be an easy whitelisting fix, but I will return to this later (TODO(newling)).

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_batch_generic_matvec() {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : i32
  %5 = arith.index_castui %0 : i32 to index
  %6 = arith.index_castui %1 : i32 to index
  %7 = arith.index_castui %2 : i32 to index
  %8 = arith.index_castui %3 : i32 to index
  %9 = arith.index_castui %4 : i32 to index
  %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%7) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
  %11 = iree_tensor_ext.dispatch.workload.ordinal %8, 0 : index
  %12 = iree_tensor_ext.dispatch.workload.ordinal %9, 1 : index
  %13 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%5) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11}
  %14 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%6) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12}
  %15 = iree_tensor_ext.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [32, 1, %11], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11} -> tensor<32x1x?xf16>
  %16 = iree_tensor_ext.dispatch.tensor.load %14, offsets = [0, 0, 0], sizes = [32, %12, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12} -> tensor<32x?x128xf16>
  %17 = tensor.empty() : tensor<32x1x128xf16>
  %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  %19 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%15, %16 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%18 : tensor<32x1x128xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %20 = arith.mulf %in, %in_0 : f16
    %21 = arith.addf %out, %20 : f16
    linalg.yield %21 : f16
  } -> tensor<32x1x128xf16>
  iree_tensor_ext.dispatch.tensor.store %19, %10, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
  return
}


//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @dynamic_batch_generic_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               partial_reduction = [0, 0, 0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1, 1], [0, 1, 2, 3]],
//  CHECK-SAME:               thread = [0, 0, 0, 8],
//  CHECK-SAME:               workgroup = [1, 1, 1, 0]

// CDNA3: LLVMGPUVectorDistribute

// -----

// This test uses special heuristics that needs to check the backend in the #hal.executable.target.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @vmt1() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<1x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<1x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  return
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @vmt1()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
//  CHECK-SAME:               partial_reduction = [0, 0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 0, 8],
//  CHECK-SAME:               workgroup = [1, 8, 0]


// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @matvec_like_no_m_dim() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4096], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096xf16>> -> tensor<4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32000xf16>) -> tensor<32000xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%3, %4 : tensor<4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<32000xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0], sizes = [32000], strides = [1] : tensor<32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000xf16>>
  return
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @matvec_like_no_m_dim()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 64], [0, 1]],
//  CHECK-SAME:               partial_reduction = [0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1], [0, 1]],
//  CHECK-SAME:               thread = [0, 8],
//  CHECK-SAME:               workgroup = [8, 0]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matvec_unit_n_dim() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x1xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<32000x1xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32000x1xf16>) -> tensor<32000x1xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %3 : tensor<32000x4096xf16>, tensor<1x4096xf16>) outs(%6 : tensor<32000x1xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<32000x1xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 1], strides = [1, 1] : tensor<32000x1xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x1xf16>>
  return
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @matvec_unit_n_dim()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
//  CHECK-SAME:               partial_reduction = [0, 0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 0, 8],
//  CHECK-SAME:               workgroup = [8, 1, 0]

// -----

// This test uses special heuristics that needs to check the backend in the #hal.executable.target.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @vmt2() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<1x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<1x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
  return
}

//   CDNA3-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [32, 1, 1] subgroup_size = 32
// CDNA3-LABEL: func.func @vmt2()
//  CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//       CDNA3:   linalg.generic
//  CDNA3-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CDNA3-SAME:               lane_basis = {{\[}}[1, 1, 32], [0, 1, 2]],
//  CDNA3-SAME:               partial_reduction = [0, 0, 256],
//  CDNA3-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CDNA3-SAME:               thread = [0, 0, 8],
//  CDNA3-SAME:               workgroup = [1, 4, 0]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
func.func @i4_dequant_matvec() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xf16>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
  %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
  %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
  %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
  %9 = tensor.empty() : tensor<4096xf16>
  %10 = tensor.empty() : tensor<4096x32x128xf16>
  %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<4096xf16>) -> tensor<4096xf16>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x32x128xi4>, tensor<4096x32xf16>, tensor<4096x32xf16>) outs(%10 : tensor<4096x32x128xf16>) {
  ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
    %14 = arith.extui %in : i4 to i32
    %15 = arith.uitofp %14 : i32 to f16
    %16 = arith.subf %15, %in_1 : f16
    %17 = arith.mulf %16, %in_0 : f16
    linalg.yield %17 : f16
  } -> tensor<4096x32x128xf16>
  %13 = linalg.generic {indexing_maps = [#map2, #map, #map3], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<32x128xf16>, tensor<4096x32x128xf16>) outs(%11 : tensor<4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %14 = arith.mulf %in, %in_0 : f16
    %15 = arith.addf %14, %out : f16
    linalg.yield %15 : f16
  } -> tensor<4096xf16>
  iree_tensor_ext.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096xf16>>
  return
}

// TODO: We should process multiple rows per subgroup.

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64
//       CHECK: func.func @i4_dequant_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
//  CHECK-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CHECK-SAME:               lane_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
//  CHECK-SAME:               partial_reduction = [0, 1, 128],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 1, 2],
//  CHECK-SAME:               workgroup = [8, 0, 0]

// -----

// Send 2xNxK mmt to the warp reduction pipeline.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @skinny_mmt() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<2x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32000xf16>) -> tensor<2x32000xf16>
  %7 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
   ins(%3, %4 : tensor<2x4096xf16>, tensor<32000x4096xf16>)
   outs(%6 : tensor<2x32000xf16>) -> tensor<2x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 32000], strides = [1, 1] : tensor<2x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32000xf16>>
  return
}

// CHECK-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Send Mx2xK mmt to the warp reduction pipeline.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @skinny_mmt() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<32000x2xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32000x2xf16>) -> tensor<32000x2xf16>
  %7 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%4, %3 : tensor<32000x4096xf16>, tensor<2x4096xf16>)
    outs(%6 : tensor<32000x2xf16>) -> tensor<32000x2xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 2], strides = [1, 1] : tensor<32000x2xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf16>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @not_vmt() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x4096xf16>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x32000xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [5, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x4096xf16>> -> tensor<5x4096xf16>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
  %5 = tensor.empty() : tensor<5x32000xf16>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<5x32000xf16>) -> tensor<5x32000xf16>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<5x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<5x32000xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.mulf %in, %in_0 : f16
    %9 = arith.addf %out, %8 : f16
    linalg.yield %9 : f16
  } -> tensor<5x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [5, 32000], strides = [1, 1] : tensor<5x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x32000xf16>>
  return
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false, no_reduce_shared_memory_bank_conflicts = true, use_igemm_convolution = false>}>
//       CHECK: func.func @not_vmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 8], thread = [1, 128, 0], workgroup = [1, 128, 1]}>

// -----


  func.func @dynamic_parallel_dims_dispatch_0_reduction_Dx4096_f16xf32() {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
    %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
    %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
    %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
    %4 = arith.extui %0 : i32 to i64
    %5 = arith.extui %1 : i32 to i64
    %6 = arith.shli %5, %c32_i64 : i64
    %7 = arith.ori %4, %6 : i64
    %8 = arith.index_castui %7 : i64 to index
    %9 = arith.extui %2 : i32 to i64
    %10 = arith.extui %3 : i32 to i64
    %11 = arith.shli %10, %c32_i64 : i64
    %12 = arith.ori %9, %11 : i64
    %13 = arith.index_castui %12 : i64 to index
    %14:2 = util.assume.int
        %8<udiv = 4>,
        %13<umin = 0, umax = 36028797018963964, udiv = 4>
      : index, index
    %15 = iree_tensor_ext.dispatch.workload.ordinal %14#0, 0 : index
    %16 = iree_tensor_ext.dispatch.workload.ordinal %14#1, 1 : index
    %17 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%16}
    %18 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%15}
    %19 = iree_tensor_ext.dispatch.tensor.load %17, offsets = [0, 0], sizes = [%16, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x4096xf16>>{%16} -> tensor<?x4096xf16>
    %20 = tensor.empty(%15) : tensor<?xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<?xf32>) -> tensor<?xf32>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%19 : tensor<?x4096xf16>) outs(%21 : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %23 = arith.extf %in : f16 to f32
      %24 = arith.addf %23, %out : f32
      linalg.yield %24 : f32
    } -> tensor<?xf32>
    iree_tensor_ext.dispatch.tensor.store %22, %18, offsets = [0], sizes = [%15], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%15}
    return
  }

//      CHECK:   #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-SAME:   workgroup_size = [512, 1, 1] subgroup_size = 64

//      CDNA:   #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CDNA-SAME:   workgroup_size = [512, 1, 1] subgroup_size = 64


// -----

func.func @test_dyn_reduction() {
  %c32 = arith.constant 32 : index
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -2.400000e+02 : f8E4M3FNUZ
  %cst_1 = arith.constant 2.400000e+02 : f8E4M3FNUZ
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
  %4 = arith.extui %0 : i32 to i64
  %5 = arith.extui %1 : i32 to i64
  %6 = arith.shli %5, %c32_i64 : i64
  %7 = arith.ori %4, %6 : i64
  %8 = arith.index_castui %7 : i64 to index
  %9 = arith.extui %2 : i32 to i64
  %10 = arith.extui %3 : i32 to i64
  %11 = arith.shli %10, %c32_i64 : i64
  %12 = arith.ori %9, %11 : i64
  %13 = arith.index_castui %12 : i64 to index
  %14:2 = util.assume.int
      %8<umin = 0, umax = 288230376151711712, udiv = 32>,
      %13<umin = 0, umax = 288230376151711712, udiv = 32>
    : index, index
  %15 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<f32>>
  %16 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf8E4M3FNUZ>>
  %17 = iree_tensor_ext.dispatch.workload.ordinal %14#0, 0 : index
  %18 = iree_tensor_ext.dispatch.workload.ordinal %14#1, 1 : index
  %19 = arith.divsi %17, %c32 : index
  %20 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32xf8E4M3FNUZ>>{%19}
  %21 = arith.divsi %18, %c32 : index
  %22 = hal.interface.binding.subspan layout(<constants = 4, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32x128xf8E4M3FNUZ>>{%21}
  %23 = iree_tensor_ext.dispatch.tensor.load %15, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
  %24 = tensor.empty() : tensor<128x128xf8E4M3FNUZ>
  %25 = tensor.empty() : tensor<128x128xf32>
  %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %27 = iree_tensor_ext.dispatch.tensor.load %20, offsets = [0, 0, 0], sizes = [128, %19, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32xf8E4M3FNUZ>>{%19} -> tensor<128x?x32xf8E4M3FNUZ>
  %28 = iree_tensor_ext.dispatch.tensor.load %22, offsets = [0, 0, 0, 0], sizes = [128, %21, 32, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x?x32x128xf8E4M3FNUZ>>{%21} -> tensor<128x?x32x128xf8E4M3FNUZ>
  %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%27, %28 : tensor<128x?x32xf8E4M3FNUZ>, tensor<128x?x32x128xf8E4M3FNUZ>) outs(%26 : tensor<128x128xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_2: f8E4M3FNUZ, %out: f32):
    %31 = arith.extf %in : f8E4M3FNUZ to f32
    %32 = arith.extf %in_2 : f8E4M3FNUZ to f32
    %33 = arith.mulf %31, %32 : f32
    %34 = arith.addf %out, %33 : f32
    linalg.yield %34 : f32
  } -> tensor<128x128xf32>
  %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %23 : tensor<128x128xf32>, tensor<f32>) outs(%24 : tensor<128x128xf8E4M3FNUZ>) {
  ^bb0(%in: f32, %in_2: f32, %out: f8E4M3FNUZ):
    %31 = arith.truncf %in : f32 to f8E4M3FNUZ
    %32 = arith.truncf %in_2 : f32 to f8E4M3FNUZ
    %33 = arith.divf %31, %32 : f8E4M3FNUZ
    %34 = arith.cmpf ult, %33, %cst_0 : f8E4M3FNUZ
    %35 = arith.select %34, %cst_0, %33 : f8E4M3FNUZ
    %36 = arith.cmpf ugt, %35, %cst_1 : f8E4M3FNUZ
    %37 = arith.select %36, %cst_1, %35 : f8E4M3FNUZ
    linalg.yield %37 : f8E4M3FNUZ
  } -> tensor<128x128xf8E4M3FNUZ>
  iree_tensor_ext.dispatch.tensor.store %30, %16, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf8E4M3FNUZ> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf8E4M3FNUZ>>
  return
}

//      CHECK: #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
// CHECK-SAME: workgroup_size = [2, 1, 1] subgroup_size = 64,
