// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-2d-scalable-to-1d-scalable{assume-arm-sme=true},cse))" --split-input-file %s | FileCheck %s

#compute_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [[4], [4]], [0, 0], [0, 0]]>
#matmul_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0], [[4], [4], 0], [0, 0, 1], [0, 0, 0]]>
#dim_0_map = affine_map<(d0)[s0] -> (-d0 + 32400, s0)>
#dim_1_map = affine_map<(d0)[s0] -> (-d0 + 16, s0)>

// Here's an example from a dispatch where a matmul has been given a 2D-scalable
// lowering config (#matmul_config) for ArmSME. That config has been propagated
// to compute ops within that same dispatch as (#compute_config).
//
// This is okay for the linalg.fill but the linalg.generic cannot be lowered
// to make use of 2D scalable vectors. ArmSME only supports 2D scalable outer
// products, so if it's not an outer product, we can only scalably vectorize in
// one dimension.
//
// The initial tile-and-fuse pass requires lowering configs to be consistent,
// so we keep the keep the lowering_configs unchanged until after that pass.
//
// 2d-scalable-to-1d-scalable can then remove unsupported scalable
// dimensions, and introduce loops. This results in dispatches that fuse both
// SME and SVE.

// Extracted from an IR dump after iree-llvmcpu-tile-and-fuse:
func.func @scalable_2d_matmul_and_generic(%arg0: tensor<32400x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<32400x16xf32>, %arg3: tensor<16xf32>) -> tensor<32400x16xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32400 = arith.constant 32400 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.vscale
  %1 = arith.muli %0, %c4 : index
  %2 = scf.for %arg4 = %c0 to %c32400 step %1 iter_args(%arg5 = %arg2) -> (tensor<32400x16xf32>) {
    %3 = scf.for %arg6 = %c0 to %c16 step %1 iter_args(%arg7 = %arg5) -> (tensor<32400x16xf32>) {
      %4 = affine.min #dim_0_map(%arg4)[%1]
      %5 = affine.min #dim_1_map(%arg6)[%1]
      %extracted_slice = tensor.extract_slice %arg0[%arg4, 0] [%4, 32] [1, 1] : tensor<32400x32xf32> to tensor<?x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg6] [32, %5] [1, 1] : tensor<32x16xf32> to tensor<32x?xf32>
      %6 = tensor.empty(%4, %5) : tensor<?x?xf32>
      %7 = linalg.fill {lowering_config = #compute_config}
        ins(%cst : f32) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %8 = linalg.matmul {lowering_config = #matmul_config}
        ins(%extracted_slice, %extracted_slice_0 : tensor<?x32xf32>, tensor<32x?xf32>)
        outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg6] [%5] [1] : tensor<16xf32> to tensor<?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg7[%arg4, %arg6] [%4, %5] [1, 1] : tensor<32400x16xf32> to tensor<?x?xf32>
      %9 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%8, %extracted_slice_1 : tensor<?x?xf32>, tensor<?xf32>)
        outs(%extracted_slice_2 : tensor<?x?xf32>) attrs =  {lowering_config = #compute_config} {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %10 = arith.mulf %in, %in_3 : f32
        linalg.yield %10 : f32
      } -> tensor<?x?xf32>
      %inserted_slice = tensor.insert_slice %9 into %arg7[%arg4, %arg6] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<32400x16xf32>
      scf.yield %inserted_slice : tensor<32400x16xf32>
    }
    scf.yield %3 : tensor<32400x16xf32>
  }
  return %2 : tensor<32400x16xf32>
}
// CHECK: #[[FILL_CONFIG:.*]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0], {{\[}}[4], [4]], [0, 0], [0, 0]]>
// CHECK: #[[MATMUL_CONFIG:.*]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], {{\[}}[4], [4], 0], [0, 0, 1], [0, 0, 0]]>
// CHECK: #[[GENERIC_CONFIG:.*]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0], [4, [4]], [0, 0], [0, 0]]>
//
//      CHECK: func.func @scalable_2d_matmul_and_generi
//      CHECK:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[VSCALE:.*]] = vector.vscale
//      CHECK:   %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
//      CHECK:   scf.for
// CHECK-SAME:    step %[[C4_VSCALE]]
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:      step %[[C4_VSCALE]]
// CHECK-SAME:     {
//      CHECK:       linalg.fill
// CHECK-SAME:         lowering_config = #[[FILL_CONFIG]]
//      CHECK:       linalg.matmul
// CHECK-SAME:         lowering_config = #[[MATMUL_CONFIG]]
//      CHECK:       scf.for
// CHECK-SAME:        step %[[C4]]
// CHECK-SAME:       {
//      CHECK:         linalg.generic
// CHECK-SAME:           lowering_config = #[[GENERIC_CONFIG]]
//      CHECK:       }
//      CHECK:     }
//      CHECK:   }

// -----

#lowering_config_parallel_only = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [[4], [4]]]>

// CHECK: #[[GENERIC_CONFIG:.*]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0], [4, [4]]]>
///
//      CHECK: func.func @should_not_crash
//      CHECK:   scf.for
//      CHECK:         linalg.generic
// CHECK-SAME:           lowering_config = #[[GENERIC_CONFIG]]
func.func @should_not_crash(%a: tensor<?x?xf32>, %b: tensor<?xf32>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%a, %b : tensor<?x?xf32>, tensor<?xf32>)
    outs(%c : tensor<?x?xf32>) attrs = {lowering_config = #lowering_config_parallel_only} {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %10 = arith.mulf %in, %in_3 : f32
    linalg.yield %10 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// In this example, both scalable flags are dropped - the affine
// map corresponding to the output is not an identity and that's currently not
// supported. This is to prevent transpositions using scalable vectors, e.g.:
//
//    vector.transpose %47, [1, 0] : vector<4x[8]xf32> to vector<[8]x4xf32>
//
// ATM, we are unable to lower such Ops to SVE/SSVE. Note - this is quite
// conservative as non-identity affine maps could also describe
// non-transpisation (that we might be able to lower).

#lowering_config_parallel_only = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [[4], [4]]]>

#map1 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: #[[GENERIC_CONFIG:.*]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0], [4, 4]]>
///
//      CHECK: func.func @generic_with_a_non_identity_out_map
//      CHECK:   scf.for
//      CHECK:      scf.for
//      CHECK:         linalg.generic
// CHECK-SAME:           lowering_config = #[[GENERIC_CONFIG]]

func.func @generic_with_a_non_identity_out_map(
    %arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>,
    %cst_0: tensor<?xf32>,
    %cst_1: tensor<?xf32>) -> tensor<?x?xf32> {
  %6 = linalg.generic {
    indexing_maps = [#map3, #map1, #map1, #map4],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %cst_0, %cst_1 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
    outs(%arg1 : tensor<?x?xf32>)  attrs = {lowering_config = #lowering_config_parallel_only} {
  ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
    %7 = arith.mulf %in, %in_2 : f32
    %8 = arith.addf %7, %in_3 : f32
    linalg.yield %8 : f32
  } -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}
