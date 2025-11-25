// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-split-reduction{enable-fp-reduction-reordering=true},cse,canonicalize))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-split-reduction,cse,canonicalize))" --split-input-file %s | FileCheck %s --check-prefix=DISABLEREASSOC

#config = #iree_cpu.lowering_config<vector_reduction = [0, 0, 0, 8]>
#config1 = #iree_cpu.lowering_config<vector_reduction = [0, 0, 0, 16]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @softmax_codegen_config(%arg0: tensor<2x5x4096x4096xf32>) -> tensor<2x5x4096x4096xf32> {
  %0 = tensor.empty() : tensor<2x5x4096x4096xf32>
  %1 = tensor.empty() : tensor<2x5x4096xf32>
  %cst = arith.constant -1.000000e+30 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x5x4096xf32>) -> tensor<2x5x4096xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<2x5x4096x4096xf32>) outs(%2 : tensor<2x5x4096xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.maximumf %in, %out : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<2x5x4096x4096xf32>, tensor<2x5x4096xf32>) outs(%0 : tensor<2x5x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.subf %in, %in_1 : f32
    %9 = math.exp %8 : f32
    linalg.yield %9 : f32
  } -> tensor<2x5x4096x4096xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %5 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<2x5x4096xf32>) -> tensor<2x5x4096xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4 : tensor<2x5x4096x4096xf32>) outs(%5 : tensor<2x5x4096xf32>) attrs =  {lowering_config = #config1} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.addf %in, %out : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %6 : tensor<2x5x4096x4096xf32>, tensor<2x5x4096xf32>) outs(%0 : tensor<2x5x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.divf %in, %in_1 : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096x4096xf32>
  return %7 : tensor<2x5x4096x4096xf32>
}
// CHECK: func.func @softmax_codegen_config
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[RES1:.+]] = linalg.generic
// CHECK:           scf.yield %[[RES1]] : tensor<1x1x1x8xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[RES2:.+]] = linalg.generic
// CHECK:           scf.yield %[[RES2]] : tensor<1x1x1x16xf32>

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [0, 0, 0, 8]>
#config1 = #iree_cpu.lowering_config<vector_reduction =  [0, 0, 0, 16]>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @softmax_cpu_config(%arg0: tensor<2x5x4096x4096xf32>) -> tensor<2x5x4096x4096xf32> {
  %0 = tensor.empty() : tensor<2x5x4096x4096xf32>
  %1 = tensor.empty() : tensor<2x5x4096xf32>
  %cst = arith.constant -1.000000e+30 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<2x5x4096xf32>) -> tensor<2x5x4096xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<2x5x4096x4096xf32>) outs(%2 : tensor<2x5x4096xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.maximumf %in, %out : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %3 : tensor<2x5x4096x4096xf32>, tensor<2x5x4096xf32>) outs(%0 : tensor<2x5x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.subf %in, %in_1 : f32
    %9 = math.exp %8 : f32
    linalg.yield %9 : f32
  } -> tensor<2x5x4096x4096xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %5 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<2x5x4096xf32>) -> tensor<2x5x4096xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4 : tensor<2x5x4096x4096xf32>) outs(%5 : tensor<2x5x4096xf32>) attrs =  {lowering_config = #config1} {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.addf %in, %out : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %6 : tensor<2x5x4096x4096xf32>, tensor<2x5x4096xf32>) outs(%0 : tensor<2x5x4096x4096xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.divf %in, %in_1 : f32
    linalg.yield %8 : f32
  } -> tensor<2x5x4096x4096xf32>
  return %7 : tensor<2x5x4096x4096xf32>
}
// CHECK: func.func @softmax_cpu_config
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[RES1:.+]] = linalg.generic
// CHECK:           scf.yield %[[RES1]] : tensor<1x1x1x8xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           %[[RES2:.+]] = linalg.generic
// CHECK:           scf.yield %[[RES2]] : tensor<1x1x1x16xf32>

// -----

// Do not split operations with indexing semantics
// See : https://github.com/iree-org/iree/issues/14934
#config = #iree_cpu.lowering_config<vector_reduction = [4]>
func.func @dont_split_with_indexing_semantics(%arg0 : tensor<4096xf32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
    ins(%arg0: tensor<4096xf32>) outs(%arg1 :tensor<f32>) attrs = {lowering_config = #config} {
    ^bb0(%b0 : f32, %b1 : f32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.addf %2, %b1 : f32
      linalg.yield %3 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL: func @dont_split_with_indexing_semantics
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       iterator_types = ["reduction"]
//       CHECK:   return %[[GENERIC]]

// -----

// check usage of result data type for respecting disable-reassociation flag.
// See https://github.com/iree-org/iree/issues/14934#issuecomment-1716552762
#config = #iree_cpu.lowering_config<vector_reduction = [4]>
func.func @dont_reassociate(%arg0 : tensor<4096xi32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
    ins(%arg0: tensor<4096xi32>) outs(%arg1 :tensor<f32>) attrs = {lowering_config = #config} {
    ^bb0(%b0 : i32, %b1 : f32):
      %2 = arith.sitofp %b0 : i32 to f32
      %3 = arith.addf %2, %b1 : f32
      linalg.yield %3 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
// DISABLEREASSOC-LABEL: func @dont_reassociate
//       DISABLEREASSOC:   %[[GENERIC:.+]] = linalg.generic
//  DISABLEREASSOC-SAME:       iterator_types = ["reduction"]
//       DISABLEREASSOC:   return %[[GENERIC]]
