// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-prepare-ukernels))" --split-input-file %s | FileCheck %s

func.func @batch_mmt4d_with_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%0 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[FILL]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>

// -----

func.func @batch_mmt4d_with_no_fill(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_no_fill
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<1x10x80x8x4xf32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xf32>

// -----

func.func @do_not_decompose_batch_mmt4d(%arg0: tensor<1x10x32x8x1xf32>, %arg1: tensor<1x80x32x4x1xf32>, %arg2: tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x10x32x8x1xf32>, tensor<1x80x32x4x1xf32>) outs(%arg2 : tensor<1x10x80x8x4xf32>) -> tensor<1x10x80x8x4xf32>
  return %1 : tensor<1x10x80x8x4xf32>
}
// CHECK-LABEL: func.func @do_not_decompose_batch_mmt4d
// CHECK:         batch_mmt4d

// -----

func.func @batch_mmt4d_with_extened_inputs(%arg0: tensor<1x10x32x8x1xi8>, %arg1: tensor<1x80x32x4x1xi8>, %arg2: tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x10x32x8x1xi32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                       ins(%arg0 : tensor<1x10x32x8x1xi8>) outs(%0 : tensor<1x10x32x8x1xi32>) {
  ^bb0(%in: i8, %out: i32):
    %6 = arith.extsi %in : i8 to i32
    linalg.yield %6 : i32
  } -> tensor<1x10x32x8x1xi32>
  %2 = tensor.empty() : tensor<1x80x32x4x1xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
                       ins(%arg1 : tensor<1x80x32x4x1xi8>) outs(%2 : tensor<1x80x32x4x1xi32>) {
  ^bb0(%in: i8, %out: i32):
    %6 = arith.extsi %in : i8 to i32
    linalg.yield %6 : i32
  } -> tensor<1x80x32x4x1xi32>
  %4 = linalg.fill ins(%c0_i32 : i32) outs(%arg2 : tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32>
  %5 = linalg.batch_mmt4d ins(%1, %3 : tensor<1x10x32x8x1xi32>, tensor<1x80x32x4x1xi32>) outs(%4 : tensor<1x10x80x8x4xi32>) -> tensor<1x10x80x8x4xi32>
  return %5 : tensor<1x10x80x8x4xi32>
}

// CHECK:      #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @batch_mmt4d_with_extened_inputs
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x10x32x8x1xi8>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x80x32x4x1xi8>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<1x10x80x8x4xi32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<1x10x80x8x4xi32> to tensor<10x80x8x4xi32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : i32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xi32>) -> tensor<10x80x8x4xi32>
// CHECK-DAG:    %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<1x10x32x8x1xi8> to tensor<10x32x8x1xi8>
// CHECK-DAG:    %[[INIT_GEN_LHS:.+]] = tensor.empty() : tensor<10x32x8x1xi32>
// CHECK-DAG:    %[[GEN_LHS:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXT_LHS]] : tensor<10x32x8x1xi8>) outs(%[[INIT_GEN_LHS]] : tensor<10x32x8x1xi32>) {
// CHECK-NEXT:       ^bb0(%[[ARGIN_GEN_LHS:.+]]: i8, %[[ARGOUT_GEN_LHS:.+]]: i32):
// CHECK-NEXT:         %[[EXT_GEN_LHS:.+]] = arith.extsi %[[ARGIN_GEN_LHS]] : i8 to i32
// CHECK-NEXT:         linalg.yield %[[EXT_GEN_LHS]] : i32
// CHECK-DAG:    %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<1x80x32x4x1xi8> to tensor<80x32x4x1xi8>
// CHECK-DAG:    %[[INIT_GEN_RHS:.+]] = tensor.empty() : tensor<80x32x4x1xi32>
// CHECK-DAG:    %[[GEN_RHS:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[EXT_RHS]] : tensor<80x32x4x1xi8>) outs(%[[INIT_GEN_RHS]] : tensor<80x32x4x1xi32>) {
// CHECK-NEXT:       ^bb0(%[[ARGIN_GEN_RHS:.+]]: i8, %[[ARGOUT_GEN_RHS:.+]]: i32):
// CHECK-NEXT:         %[[EXT_GEN_RHS:.+]] = arith.extsi %[[ARGIN_GEN_RHS]] : i8 to i32
// CHECK-NEXT:         linalg.yield %[[EXT_GEN_RHS]] : i32
// CHECK:        %[[MMT4D:.+]] = linalg.mmt4d ins(%[[GEN_LHS]], %[[GEN_RHS]] : tensor<10x32x8x1xi32>, tensor<80x32x4x1xi32>) outs(%[[FILL]] : tensor<10x80x8x4xi32>) -> tensor<10x80x8x4xi32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUT]][0, 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xi32> into tensor<1x10x80x8x4xi32>
// CHECK:        return %[[INS]] : tensor<1x10x80x8x4xi32>

// -----

func.func @batch_mmt4d_with_fill_batch_dim(%arg0: tensor<12x10x32x8x1xf32>, %arg1: tensor<12x80x32x4x1xf32>, %arg2: tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32>
  %1 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<12x10x32x8x1xf32>, tensor<12x80x32x4x1xf32>) outs(%0 : tensor<12x10x80x8x4xf32>) -> tensor<12x10x80x8x4xf32>
  return %1 : tensor<12x10x80x8x4xf32>
}

// CHECK:      func.func @batch_mmt4d_with_fill_batch_dim
// CHECK-SAME:   %[[LHS:.+]]: tensor<12x10x32x8x1xf32>,
// CHECK-SAME:   %[[RHS:.+]]: tensor<12x80x32x4x1xf32>,
// CHECK-SAME:   %[[OUT:.+]]: tensor<12x10x80x8x4xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C12:.+]] = arith.constant 12 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[TILED_RES:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C12]] step %[[C1]] iter_args(%[[OUTPUT:.+]] = %[[OUT]]) -> (tensor<12x10x80x8x4xf32>) {
// CHECK-DAG:      %[[EXT_OUT:.+]] = tensor.extract_slice %[[OUTPUT]][%[[IV]], 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<12x10x80x8x4xf32> to tensor<10x80x8x4xf32>
// CHECK:          %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EXT_OUT]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK-DAG:      %[[EXT_LHS:.+]] = tensor.extract_slice %[[LHS]][%[[IV]], 0, 0, 0, 0] [1, 10, 32, 8, 1] [1, 1, 1, 1, 1] : tensor<12x10x32x8x1xf32> to tensor<10x32x8x1xf32>
// CHECK-DAG:      %[[EXT_RHS:.+]] = tensor.extract_slice %[[RHS]][%[[IV]], 0, 0, 0, 0] [1, 80, 32, 4, 1] [1, 1, 1, 1, 1] : tensor<12x80x32x4x1xf32> to tensor<80x32x4x1xf32>
// CHECK:          %[[MMT4D:.+]] = linalg.mmt4d ins(%[[EXT_LHS]], %[[EXT_RHS]] : tensor<10x32x8x1xf32>, tensor<80x32x4x1xf32>) outs(%[[FILL]] : tensor<10x80x8x4xf32>) -> tensor<10x80x8x4xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MMT4D]] into %[[OUTPUT]][%[[IV]], 0, 0, 0, 0] [1, 10, 80, 8, 4] [1, 1, 1, 1, 1] : tensor<10x80x8x4xf32> into tensor<12x10x80x8x4xf32>
// CHECK:          scf.yield %[[INS]] : tensor<12x10x80x8x4xf32>
// CHECK:        }
// CHECK:        return %[[TILED_RES]] : tensor<12x10x80x8x4xf32>

// -----

func.func @batch_mmt4d_with_lowering_config(%arg0: tensor<12x4x64x8x1xf16>, %arg1: tensor<12x4x64x8x1xf16>, %arg2: tensor<12x4x4x8x8xf16>) -> tensor<12x4x4x8x8xf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 0, 0], [1, 1, 1, 0, 8], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]>} ins(%cst : f16) outs(%arg2 : tensor<12x4x4x8x8xf16>) -> tensor<12x4x4x8x8xf16>
  %1 = linalg.batch_mmt4d {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 4, 4, 0, 0, 0, 0], [1, 1, 1, 0, 8, 8, 0], [0, 0, 0, 1, 0, 0, 1]]>} ins(%arg0, %arg1 : tensor<12x4x64x8x1xf16>, tensor<12x4x64x8x1xf16>) outs(%0 : tensor<12x4x4x8x8xf16>) -> tensor<12x4x4x8x8xf16>
  return %1 : tensor<12x4x4x8x8xf16>
}
// CHECK:      #[[CONFIG1:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 4, 0, 0], [1, 1, 0, 8], [0, 0, 0, 0], [0, 0, 0, 0]]>
// CHECK:      #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 4, 0, 0, 0, 0], [1, 1, 0, 8, 8, 0], [0, 0, 1, 0, 0, 1]]>
// CHECK:      func.func @batch_mmt4d_with_lowering_config
// CHECK:        linalg.fill {lowering_config = #[[CONFIG1]]}
// CHECK:        linalg.mmt4d {lowering_config = #[[CONFIG2]]}

// -----

func.func @pack_without_outer_dims_perm(%arg0: tensor<1x16384x512xbf16>, %arg1: tensor<1x1024x256x16x2xbf16>) -> tensor<1x1024x256x16x2xbf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "pack", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : bf16
  %pack = linalg.pack %arg0 inner_dims_pos = [1, 2] inner_tiles = [16, 2] into %arg1 : tensor<1x16384x512xbf16> -> tensor<1x1024x256x16x2xbf16>
  return %pack : tensor<1x1024x256x16x2xbf16>
}
// CHECK:      func.func @pack_without_outer_dims_perm
// CHECK-SAME:   %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:   %[[DEST:[0-9a-zA-Z]+]]
// CHECK:        %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:     tensor<1x16384x512xbf16> to tensor<16384x512xbf16>
// CHECK:        %[[DEST_SLICE:.+]] = tensor.extract_slice %[[DEST]]
// CHECK-SAME:      tensor<1x1024x256x16x2xbf16> to tensor<1024x256x16x2xbf16>
// CHECK:        %[[PACK:.+]] = linalg.pack %[[SRC_SLICE]]
// CHECK-SAME:     inner_dims_pos = [0, 1] inner_tiles = [16, 2]
// CHECK-SAME:     into %[[DEST_SLICE]]

// -----

func.func @pack_with_outer_dims_perm(%arg0: tensor<484x16x64xbf16>, %arg1: tensor<64x31x8x16x2xbf16>) -> tensor<64x31x8x16x2xbf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "pack", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : bf16
  %pack = linalg.pack %arg0 padding_value(%cst : bf16) outer_dims_perm = [2, 0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %arg1 : tensor<484x16x64xbf16> -> tensor<64x31x8x16x2xbf16>
  return %pack : tensor<64x31x8x16x2xbf16>
}
// CHECK:      func.func @pack_with_outer_dims_perm
// CHECK-SAME:   %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:   %[[DEST:[0-9a-zA-Z]+]]
// CHECK-DAG:    %[[PAD_VAL:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:        %[[RES:.+]] = scf.for {{.+}} iter_args(%[[ITER:.+]] = %[[DEST]]) -> (tensor<64x31x8x16x2xbf16>)
// CHECK:          %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:       tensor<484x16x64xbf16> to tensor<484x16xbf16>
// CHECK:          %[[DEST_SLICE:.+]] = tensor.extract_slice %[[ITER]]
// CHECK-SAME:       tensor<64x31x8x16x2xbf16> to tensor<31x8x16x2xbf16>
// CHECK:          %[[PACK:.+]] = linalg.pack %[[SRC_SLICE]]
// CHECK-SAME:       padding_value(%[[PAD_VAL]] : bf16)
// CHECK-SAME:       outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 2]
// CHECK-SAME:       into %[[DEST_SLICE]]
// CHECK:        return %[[RES]]

// -----

func.func @do_not_decompose_pack(%arg0: tensor<1x16384x512xbf16>, %arg1: tensor<1x1024x256x16x2xbf16>) -> tensor<1x1024x256x16x2xbf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : bf16
  %pack = linalg.pack %arg0 inner_dims_pos = [1, 2] inner_tiles = [16, 2] into %arg1 : tensor<1x16384x512xbf16> -> tensor<1x1024x256x16x2xbf16>
  return %pack : tensor<1x1024x256x16x2xbf16>
}
// CHECK-LABEL: func.func @do_not_decompose_pack
// CHECK:         linalg.pack {{.+}} : tensor<1x16384x512xbf16> -> tensor<1x1024x256x16x2xbf16>

// -----

func.func @unpack_without_transpose(%arg0: tensor<1828x8x64x16x16xf32>) -> tensor<1828x128x1024xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "unpack", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %6 = tensor.empty() : tensor<1828x128x1024xf32>
  %unpack = linalg.unpack %arg0
      outer_dims_perm = [0, 1, 2]
      inner_dims_pos = [1, 2]
      inner_tiles = [16, 16]
      into %6 : tensor<1828x8x64x16x16xf32> -> tensor<1828x128x1024xf32>
  return %unpack : tensor<1828x128x1024xf32>
}
// CHECK-LABEL:   func.func @unpack_without_transpose(
// CHECK:                                        %[[SRC:.*]]: tensor<1828x8x64x16x16xf32>) -> tensor<1828x128x1024xf32>
// CHECK:           %[[CST_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CST_1828:.*]] = arith.constant 1828 : index
// CHECK:           %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DEST:.*]] = tensor.empty() : tensor<1828x128x1024xf32>
// CHECK:           %[[RES:.*]] = scf.for %[[ITER:.*]] = %[[CST_0]] to %[[CST_1828]]
// CHECK-SAME:        step %[[CST_1]] iter_args(%[[ITER_ARG:.*]] = %[[DEST]]) -> (tensor<1828x128x1024xf32>) {
// CHECK:             %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][%[[ITER]], 0, 0, 0, 0] [1, 8, 64, 16, 16] [1, 1, 1, 1, 1]
// CHECK-SAME:          : tensor<1828x8x64x16x16xf32> to tensor<8x64x16x16xf32>
// CHECK:             %[[DEST_SLICE:.*]] = tensor.extract_slice %[[ITER_ARG]][%[[ITER]], 0, 0] [1, 128, 1024] [1, 1, 1]
// CHECK-SAME:          : tensor<1828x128x1024xf32> to tensor<128x1024xf32>
// CHECK:             %[[UNPACK:.*]] = linalg.unpack %[[SRC_SLICE]]
// CHECK-SAME:         outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:         into %[[DEST_SLICE]] : tensor<8x64x16x16xf32> -> tensor<128x1024xf32>
// CHECK:             %[[NEW_ITER_ARG:.*]] = tensor.insert_slice %[[UNPACK]] into %[[ITER_ARG]][%[[ITER]], 0, 0] [1, 128, 1024] [1, 1, 1]
// CHECK-SAME:          : tensor<128x1024xf32> into tensor<1828x128x1024xf32>
// CHECK:             scf.yield %[[NEW_ITER_ARG]] : tensor<1828x128x1024xf32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<1828x128x1024xf32>
// CHECK:         }

// -----

func.func @unpack_outer_dim_transpose(%arg0: tensor<4x8x29241x16x16xf32>) -> tensor<29241x128x64xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "unpack", target_triple="x86_64-xyz-xyz", cpu_features=""}>
} {
  %cst = arith.constant 0.000000e+00 : bf16
  %4 = tensor.empty() : tensor<29241x128x64xf32>
  %unpack = linalg.unpack %arg0 outer_dims_perm = [2, 1, 0] inner_dims_pos = [1, 2] inner_tiles = [16, 16] into %4 : tensor<4x8x29241x16x16xf32> -> tensor<29241x128x64xf32>
  return %unpack : tensor<29241x128x64xf32>
}
// CHECK-LABEL:   func.func @unpack_outer_dim_transpose(
// CHECK:                                           %[[SRC:.*]]: tensor<4x8x29241x16x16xf32>) -> tensor<29241x128x64xf32>
// CHECK:           %[[CST_1:.*]] = arith.constant 1 : index
// CHECK:           %[[CST_29K:.*]] = arith.constant 29241 : index
// CHECK:           %[[CST_0:.*]] = arith.constant 0 : index
// CHECK:           %[[DEST:.*]] = tensor.empty() : tensor<29241x128x64xf32>
// CHECK:           %[[RES:.*]] = scf.for %[[ITER:.*]] = %[[CST_0]] to %[[CST_29K]] step %[[CST_1]]
// CHECK-SAME:        iter_args(%[[ITER_ARG:.*]] = %[[DEST]]) -> (tensor<29241x128x64xf32>) {
// CHECK:             %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, 0, %[[ITER]], 0, 0] [4, 8, 1, 16, 16] [1, 1, 1, 1, 1]
// CHECK-SAME:          : tensor<4x8x29241x16x16xf32> to tensor<4x8x16x16xf32>
// CHECK:             %[[DEST_SLICE:.*]] = tensor.extract_slice %[[ITER_ARG]][%[[ITER]], 0, 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:          : tensor<29241x128x64xf32> to tensor<128x64xf32>
// CHECK:             %[[UNPACK:.*]] = linalg.unpack %[[SRC_SLICE]]
// CHECK-SAME:         outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
// CHECK-SAME:         into %[[DEST_SLICE]] : tensor<4x8x16x16xf32> -> tensor<128x64xf32>
// CHECK:             %[[NEW_ITER_ARG:.*]] = tensor.insert_slice %[[UNPACK]] into %[[ITER_ARG]][%[[ITER]], 0, 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:         : tensor<128x64xf32> into tensor<29241x128x64xf32>
// CHECK:             scf.yield %[[NEW_ITER_ARG]] : tensor<29241x128x64xf32>
// CHECK:           }
// CHECK:           return %[[RES]] : tensor<29241x128x64xf32>
// CHECK:         }
