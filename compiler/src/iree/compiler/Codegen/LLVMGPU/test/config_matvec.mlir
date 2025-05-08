// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s --check-prefix=CDNA3

#pipeline_layout = #hal.pipeline.layout<constants = 5, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_batch_matvec() {
  %c32_i64 = arith.constant 32 : i64
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
  %19 = linalg.batch_matmul ins(%15, %16 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%18 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  iree_tensor_ext.dispatch.tensor.store %19, %10, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
  return
}

//   CDNA3-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 32]{{\]}}>
//   CDNA3-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [32, 1, 1] subgroup_size = 32>
// CDNA3-LABEL: func.func @dynamic_batch_matvec()
//  CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//       CDNA3:   linalg.batch_matmul
//  CDNA3-SAME:       lowering_config = #[[$CONFIG]]

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
//  CHECK-SAME:               partial_reduction = [0, 0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 0, 8], thread_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
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
//  CHECK-SAME:               partial_reduction = [0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1], [0, 1]],
//  CHECK-SAME:               thread = [0, 8], thread_basis = {{\[}}[1, 64], [0, 1]],
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
//  CHECK-SAME:               partial_reduction = [0, 0, 512],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 0, 8], thread_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
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

//   CDNA3-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 32
// CDNA3-LABEL: func.func @vmt2()
//  CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//       CDNA3:   linalg.generic
//  CDNA3-SAME:    attrs =  {lowering_config = #iree_gpu.lowering_config<{
//  CDNA3-SAME:               partial_reduction = [0, 0, 512],
//  CDNA3-SAME:               subgroup_basis = {{\[}}[1, 1, 2], [0, 1, 2]],
//  CDNA3-SAME:               thread = [0, 0, 8], thread_basis = {{\[}}[1, 1, 32], [0, 1, 2]],
//  CDNA3-SAME:               workgroup = [1, 8, 0]

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
//  CHECK-SAME:               partial_reduction = [0, 1, 128],
//  CHECK-SAME:               subgroup_basis = {{\[}}[1, 1, 1], [0, 1, 2]],
//  CHECK-SAME:               thread = [0, 1, 2], thread_basis = {{\[}}[1, 1, 64], [0, 1, 2]],
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
  %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<2x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<2x32000xf16>) -> tensor<2x32000xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 32000], strides = [1, 1] : tensor<2x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32000xf16>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul_transpose_b
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
  %7 = linalg.matmul_transpose_b ins(%4, %3 : tensor<32000x4096xf16>, tensor<2x4096xf16>) outs(%6 : tensor<32000x2xf16>) -> tensor<32000x2xf16>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 2], strides = [1, 1] : tensor<32000x2xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32000x2xf16>>
  return
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul_transpose_b
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

func.func @dynamic_parallel_dims(%dynsize : index, %input : tensor<4x?x4096xf16>) -> tensor<4x?xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%dynsize) : tensor<4x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x?xf32>) -> tensor<4x?xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input : tensor<4x?x4096xf16>) outs(%1 : tensor<4x?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      %4 = arith.addf %3, %out : f32
      linalg.yield %4 : f32
    } -> tensor<4x?xf32>
  return %2 : tensor<4x?xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 64]{{\]}}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//      CHECK: func @dynamic_parallel_dims
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

//  CDNA3-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 32]{{\]}}
//  CDNA3-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [32, 1, 1] subgroup_size = 32>
//      CDNA3: func @dynamic_parallel_dims
// CDNA3-SAME:     translation_info = #[[TRANSLATION]]
//      CDNA3:   linalg.generic
// CDNA3-SAME:       lowering_config = #[[CONFIG]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
func.func @test_dyn_reduction(%arg0: tensor<128x?x32xf8E4M3FNUZ>, %arg1: tensor<128x?x32x128xf8E4M3FNUZ>, %arg2: tensor<f32>) -> tensor<128x128xf8E4M3FNUZ> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -2.400000e+02 : f8E4M3FNUZ
  %cst_1 = arith.constant 2.400000e+02 : f8E4M3FNUZ
  %0 = tensor.empty() : tensor<128x128xf8E4M3FNUZ>
  %1 = tensor.empty() : tensor<128x128xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<128x?x32xf8E4M3FNUZ>, tensor<128x?x32x128xf8E4M3FNUZ>) outs(%2 : tensor<128x128xf32>) {
  ^bb0(%in: f8E4M3FNUZ, %in_2: f8E4M3FNUZ, %out: f32):
    %5 = arith.extf %in : f8E4M3FNUZ to f32
    %6 = arith.extf %in_2 : f8E4M3FNUZ to f32
    %7 = arith.mulf %5, %6 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<128x128xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%3, %arg2 : tensor<128x128xf32>, tensor<f32>) outs(%0 : tensor<128x128xf8E4M3FNUZ>) {
  ^bb0(%in: f32, %in_2: f32, %out: f8E4M3FNUZ):
    %5 = arith.truncf %in : f32 to f8E4M3FNUZ
    %6 = arith.truncf %in_2 : f32 to f8E4M3FNUZ
    %7 = arith.divf %5, %6 : f8E4M3FNUZ
    %8 = arith.cmpf ult, %7, %cst_0 : f8E4M3FNUZ
    %9 = arith.select %8, %cst_0, %7 : f8E4M3FNUZ
    %10 = arith.cmpf ugt, %9, %cst_1 : f8E4M3FNUZ
    %11 = arith.select %10, %cst_1, %9 : f8E4M3FNUZ
    linalg.yield %11 : f8E4M3FNUZ
  } -> tensor<128x128xf8E4M3FNUZ>
  return %4 : tensor<128x128xf8E4M3FNUZ>
}
//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 1, 64]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @test_dyn_reduction
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
