// RUN: iree-opt --split-input-file --iree-gpu-test-target=cdna2@vulkan --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-generalize-named-ops),iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// Note: above we generalize named ops before selecting the lowering strategy, as selection assumes that some named ops like linalg.matmul have been generalized.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
func.func @i4_dequant_matvec_f32(%5: tensor<4096x86x128xi4>, %6: tensor<4096x86xf32>, %7: tensor<4096x86xf32>, %8: tensor<86x128xf32>) -> tensor<4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %9 = tensor.empty() : tensor<4096xf32>
  %10 = tensor.empty() : tensor<4096x86x128xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4096xf32>) -> tensor<4096xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%10 : tensor<4096x86x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %14 = arith.extui %in : i4 to i32
    %15 = arith.uitofp %14 : i32 to f32
    %16 = arith.subf %15, %in_1 : f32
    %17 = arith.mulf %16, %in_0 : f32
    linalg.yield %17 : f32
  } -> tensor<4096x86x128xf32>
  %13 = linalg.generic {indexing_maps = [#map2, #map, #map3], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%11 : tensor<4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %14 = arith.mulf %in, %in_0 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<4096xf32>
  return %13 : tensor<4096xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
func.func @i4_dequant_matvec_f32(%5: tensor<4096x32x128xi4>, %6: tensor<4096x32x1xf32>, %7: tensor<4096x32x1xf32>, %8: tensor<1x1x32x128xf32>) -> tensor<1x1x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %9 = tensor.empty() : tensor<1x1x4096xf32>
  %10 = tensor.empty() : tensor<4096x32x128xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x32x128xi4>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%10 : tensor<4096x32x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %14 = arith.extui %in : i4 to i32
    %15 = arith.uitofp %14 : i32 to f32
    %16 = arith.subf %15, %in_1 : f32
    %17 = arith.mulf %16, %in_0 : f32
    linalg.yield %17 : f32
  } -> tensor<4096x32x128xf32>
  %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%11 : tensor<1x1x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %14 = arith.mulf %in, %in_0 : f32
    %15 = arith.addf %14, %out : f32
    linalg.yield %15 : f32
  } -> tensor<1x1x4096xf32>
  return %13 : tensor<1x1x4096xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 4, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [128, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @i4_dequant_matvec_f32(%30: index, %33: tensor<4096x86x128xi4>, %34: tensor<4096x86xf32>, %35: tensor<4096x86xf32>, %36: tensor<?x86x128xf32>) -> tensor<?x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty(%30) : tensor<?x4096xf32>
  %38 = tensor.empty() : tensor<4096x86x128xf32>
  %39 = linalg.fill ins(%cst : f32) outs(%37 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  %40 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %34, %35 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%38 : tensor<4096x86x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %42 = arith.extui %in : i4 to i32
    %43 = arith.uitofp %42 : i32 to f32
    %44 = arith.subf %43, %in_1 : f32
    %45 = arith.mulf %44, %in_0 : f32
    linalg.yield %45 : f32
  } -> tensor<4096x86x128xf32>
  %41 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%36, %40 : tensor<?x86x128xf32>, tensor<4096x86x128xf32>) outs(%39 : tensor<?x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %42 = arith.mulf %in, %in_0 : f32
    %43 = arith.addf %42, %out : f32
    linalg.yield %43 : f32
  } -> tensor<?x4096xf32>
  return %41 : tensor<?x4096xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f32(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
func.func @i4_dequant_matvec_f16(%5: tensor<4096x86x128xi4>, %6: tensor<4096x86x1xf16>, %7: tensor<4096x86x1xf16>, %8: tensor<1x1x86x128xf16>) -> tensor<1x1x4096xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %9 = tensor.empty() : tensor<1x1x4096xf16>
  %10 = tensor.empty() : tensor<4096x86x128xf16>
  %11 = linalg.fill ins(%cst : f16) outs(%9 : tensor<1x1x4096xf16>) -> tensor<1x1x4096xf16>
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86x1xf16>, tensor<4096x86x1xf16>) outs(%10 : tensor<4096x86x128xf16>) {
  ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
    %14 = arith.extui %in : i4 to i32
    %15 = arith.uitofp %14 : i32 to f16
    %16 = arith.subf %15, %in_1 : f16
    %17 = arith.mulf %16, %in_0 : f16
    linalg.yield %17 : f16
  } -> tensor<4096x86x128xf16>
  %13 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<1x1x86x128xf16>, tensor<4096x86x128xf16>) outs(%11 : tensor<1x1x4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %14 = arith.mulf %in, %in_0 : f16
    %15 = arith.addf %14, %out : f16
    linalg.yield %15 : f16
  } -> tensor<1x1x4096xf16>
  return %13 : tensor<1x1x4096xf16>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec_f16(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @i4_dequant_matvec(%30: index, %33: tensor<4096x86x128xi4>, %34: tensor<4096x86xf32>, %35: tensor<4096x86xf32>, %36: tensor<?x86x128xf32>) -> tensor<?x4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %37 = tensor.empty(%30) : tensor<?x4096xf32>
  %38 = tensor.empty() : tensor<4096x86x128xf32>
  %39 = linalg.fill ins(%cst : f32) outs(%37 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  %40 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %34, %35 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%38 : tensor<4096x86x128xf32>) {
  ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
    %42 = arith.extui %in : i4 to i32
    %43 = arith.uitofp %42 : i32 to f32
    %44 = arith.subf %43, %in_1 : f32
    %45 = arith.mulf %44, %in_0 : f32
    linalg.yield %45 : f32
  } -> tensor<4096x86x128xf32>
  %41 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%36, %40 : tensor<?x86x128xf32>, tensor<4096x86x128xf32>) outs(%39 : tensor<?x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %42 = arith.mulf %in, %in_0 : f32
    %43 = arith.addf %42, %out : f32
    linalg.yield %43 : f32
  } -> tensor<?x4096xf32>
  return %41 : tensor<?x4096xf32>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @i4_dequant_matvec(%23: index, %26: tensor<11008x32x128xi4>, %27: tensor<11008x32xf16>, %28: tensor<11008x32xf16>, %29: tensor<?x32x128xf16>) -> tensor<?x11008xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %30 = tensor.empty() : tensor<11008x32x128xf16>
  %31 = tensor.empty(%23) : tensor<?x11008xf16>
  %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x11008xf16>) -> tensor<?x11008xf16>
  %33 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %27, %28 : tensor<11008x32x128xi4>, tensor<11008x32xf16>, tensor<11008x32xf16>) outs(%30 : tensor<11008x32x128xf16>) {
  ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
    %35 = arith.extui %in : i4 to i32
    %36 = arith.uitofp %35 : i32 to f16
    %37 = arith.subf %36, %in_1 : f16
    %38 = arith.mulf %37, %in_0 : f16
    linalg.yield %38 : f16
  } -> tensor<11008x32x128xf16>
  %34 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%29, %33 : tensor<?x32x128xf16>, tensor<11008x32x128xf16>) outs(%32 : tensor<?x11008xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %35 = arith.mulf %in, %in_0 : f16
    %36 = arith.addf %35, %out : f16
    linalg.yield %36 : f16
  } -> tensor<?x11008xf16>
  return %34 : tensor<?x11008xf16>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 4, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @i4_dequant_matvec(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]

// -----

func.func @dynamic_batch_matvec(%15: tensor<32x1x?xf16>, %16: tensor<32x?x128xf16>) -> tensor<32x1x128xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %17 = tensor.empty() : tensor<32x1x128xf16>
  %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  %19 = linalg.batch_matmul ins(%15, %16 : tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%18 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
  return %19 : tensor<32x1x128xf16>
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 64]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVSubgroupReduce workgroup_size = [64, 1, 1]>
//       CHECK: func.func @dynamic_batch_matvec(
//  CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//       CHECK:   linalg.generic
//  CHECK-SAME:       lowering_config = #[[$CONFIG]]
