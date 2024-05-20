// RUN: iree-opt --split-input-file --iree-codegen-test-target=gfx940 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-codegen-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s --check-prefix=CDNA3

module {
  func.func @dynamic_batch_matvec() {
    %c32_i64 = arith.constant 32 : i64
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.constant.load[1] : i32
    %2 = hal.interface.constant.load[2] : i32
    %3 = hal.interface.constant.load[3] : i32
    %4 = hal.interface.constant.load[4] : i32
    %5 = arith.index_castui %0 : i32 to index
    %6 = arith.index_castui %1 : i32 to index
    %7 = arith.index_castui %2 : i32 to index
    %8 = arith.index_castui %3 : i32 to index
    %9 = arith.index_castui %4 : i32 to index
    %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
    %11 = flow.dispatch.workload.ordinal %8, 0 : index
    %12 = flow.dispatch.workload.ordinal %9, 1 : index
    %13 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11}
    %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12}
    %15 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0], sizes = [32, 1, %11], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x1x?xf16>>{%11} -> tensor<32x1x?xf16>
    %16 = flow.dispatch.tensor.load %14, offsets = [0, 0, 0], sizes = [32, %12, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x?x128xf16>>{%12} -> tensor<32x?x128xf16>
    %17 = tensor.empty() : tensor<32x1x128xf16>
    %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
    %19 = linalg.batch_matmul ins(%15, %16 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%18 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
    flow.dispatch.tensor.store %19, %10, offsets = [0, 0, 0], sizes = [32, 1, 128], strides = [1, 1, 1] : tensor<32x1x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x1x128xf16>>
    return
  }
}

//   CDNA3-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 32]{{\]}}>
//   CDNA3-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [32, 1, 1]>
// CDNA3-LABEL: func.func @dynamic_batch_matvec()
//  CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//       CDNA3:   linalg.batch_matmul
//  CDNA3-SAME:       lowering_config = #[[$CONFIG]]

// -----

// This test uses special heuristics that needs to check the backend in the #hal.executable.target.
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @vmt1() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
    %5 = tensor.empty() : tensor<1x32000xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %8 = arith.mulf %in, %in_0 : f16
      %9 = arith.addf %out, %8 : f16
      linalg.yield %9 : f16
    } -> tensor<1x32000xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8], [0, 0, 512]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
// CHECK-LABEL: func.func @vmt1()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// This test uses special heuristics that needs to check the backend in the #hal.executable.target.
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @vmt2() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
    %5 = tensor.empty() : tensor<1x32000xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %8 = arith.mulf %in, %in_0 : f16
      %9 = arith.addf %out, %8 : f16
      linalg.yield %9 : f16
    } -> tensor<1x32000xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
    return
  }
}

//   CDNA3-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8], [0, 0, 512]{{\]}}>
//   CDNA3-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 32>
// CDNA3-LABEL: func.func @vmt2()
//  CDNA3-SAME:     translation_info = #[[$TRANSLATION]]
//       CDNA3:   linalg.generic
//  CDNA3-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @i4_dequant_matvec() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x128xf16>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32xf16>> -> tensor<4096x32xf16>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x128xf16>> -> tensor<32x128xf16>
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
    flow.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<4096xf16>>
    return
  }
}

// TODO: We should process multiple rows per subgroup.

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 4, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @i4_dequant_matvec()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Send 2xNxK mmt to the warp reduction pipeline.

module {
  func.func @skinny_mmt() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32000xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
    %5 = tensor.empty() : tensor<2x32000xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32000xf16>) -> tensor<2x32000xf16>
    %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<2x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<2x32000xf16>) -> tensor<2x32000xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 32000], strides = [1, 1] : tensor<2x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x32000xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul_transpose_b
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

// Send Mx2xK mmt to the warp reduction pipeline.

module {
  func.func @skinny_mmt() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32000x2xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096xf16>> -> tensor<2x4096xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
    %5 = tensor.empty() : tensor<32000x2xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32000x2xf16>) -> tensor<32000x2xf16>
    %7 = linalg.matmul_transpose_b ins(%4, %3 : tensor<32000x4096xf16>, tensor<2x4096xf16>) outs(%6 : tensor<32000x2xf16>) -> tensor<32000x2xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32000, 2], strides = [1, 1] : tensor<32000x2xf16> -> !flow.dispatch.tensor<writeonly:tensor<32000x2xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUWarpReduction workgroup_size = [64, 1, 1] subgroup_size = 64>
//       CHECK: func.func @skinny_mmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.matmul_transpose_b
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @not_vmt() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5x4096xf16>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<5x32000xf16>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [5, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<5x4096xf16>> -> tensor<5x4096xf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
    %5 = tensor.empty() : tensor<5x32000xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<5x32000xf16>) -> tensor<5x32000xf16>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<5x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<5x32000xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %8 = arith.mulf %in, %in_0 : f16
      %9 = arith.addf %out, %8 : f16
      linalg.yield %9 : f16
    } -> tensor<5x32000xf16>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [5, 32000], strides = [1, 1] : tensor<5x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<5x32000xf16>>
    return
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 128, 8]{{\]}}>
//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUMatmulSimt workgroup_size = [32, 1, 1] subgroup_size = 64, {pipeline_depth = 0 : i64, store_stage = 1 : i64}>
//       CHECK: func.func @not_vmt()
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
