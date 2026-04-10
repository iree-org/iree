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

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [16]>
func.func @split_bounded_dynamic_reduction(%arg0 : tensor<256xi32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %0 = scf.for %iv = %c0 to %c256 step %c128 iter_args(%acc = %arg1) -> (tensor<i32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 256, 128)>(%iv)
    %slice = tensor.extract_slice %arg0[%iv] [%size] [1] : tensor<256xi32> to tensor<?xi32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
      ins(%slice : tensor<?xi32>) outs(%acc : tensor<i32>) attrs = {lowering_config = #config} {
      ^bb0(%in : i32, %out : i32):
        %sum = arith.addi %in, %out : i32
        linalg.yield %sum : i32
    } -> tensor<i32>
    scf.yield %reduced : tensor<i32>
  }
  return %0 : tensor<i32>
}
// CHECK-LABEL: func.func @split_bounded_dynamic_reduction
// CHECK:         %[[SIZE:.+]] = affine.min
// CHECK:         %[[OUTER:.+]] = affine.apply
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape {{.*}} output_shape [%[[OUTER]], 16]
// CHECK:         %[[INIT:.+]] = linalg.fill
// CHECK:         %[[DIM:.+]] = tensor.dim %[[EXPANDED]]
// CHECK:         %[[PARTIAL_LOOP:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[INIT]]) -> (tensor<16xi32>)
// CHECK:           %[[PARTIAL:.+]] = linalg.generic
// CHECK-SAME:        iterator_types = ["reduction", "parallel"]
// CHECK:           scf.yield %[[PARTIAL]]
// CHECK:         %[[FINAL:.+]] = linalg.generic
// CHECK-SAME:        iterator_types = ["reduction"]
// CHECK:         scf.yield %[[FINAL]]

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [16]>
func.func @dont_split_unproven_dynamic_reduction(%arg0 : tensor<250xi32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c125 = arith.constant 125 : index
  %c250 = arith.constant 250 : index
  %0 = scf.for %iv = %c0 to %c250 step %c125 iter_args(%acc = %arg1) -> (tensor<i32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 250, 125)>(%iv)
    %slice = tensor.extract_slice %arg0[%iv] [%size] [1] : tensor<250xi32> to tensor<?xi32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
      ins(%slice : tensor<?xi32>) outs(%acc : tensor<i32>) attrs = {lowering_config = #config} {
      ^bb0(%in : i32, %out : i32):
        %sum = arith.addi %in, %out : i32
        linalg.yield %sum : i32
    } -> tensor<i32>
    scf.yield %reduced : tensor<i32>
  }
  return %0 : tensor<i32>
}
// CHECK-LABEL: func.func @dont_split_unproven_dynamic_reduction
// CHECK-NOT:     tensor.expand_shape
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK:         scf.yield %[[GENERIC]]

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [16]>
func.func @split_bounded_dynamic_maximumf(%arg0 : tensor<256xf32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %0 = scf.for %iv = %c0 to %c256 step %c128 iter_args(%acc = %arg1) -> (tensor<f32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 256, 128)>(%iv)
    %slice = tensor.extract_slice %arg0[%iv] [%size] [1] : tensor<256xf32> to tensor<?xf32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
      ins(%slice : tensor<?xf32>) outs(%acc : tensor<f32>) attrs = {lowering_config = #config} {
      ^bb0(%in : f32, %out : f32):
        %max = arith.maximumf %in, %out : f32
        linalg.yield %max : f32
    } -> tensor<f32>
    scf.yield %reduced : tensor<f32>
  }
  return %0 : tensor<f32>
}
// CHECK-LABEL: func.func @split_bounded_dynamic_maximumf
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<16xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[INIT]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK:         %[[PARTIAL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction", "parallel"]
// CHECK:             %[[MAX0:.+]] = arith.maximumf
// CHECK:         %[[FINAL:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK:             %[[MAX1:.+]] = arith.maximumf

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [16]>
func.func @split_bounded_dynamic_with_linalg_index(%arg0 : tensor<256xf32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %0 = scf.for %iv = %c0 to %c256 step %c128 iter_args(%acc = %arg1) -> (tensor<i32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 256, 128)>(%iv)
    %slice = tensor.extract_slice %arg0[%iv] [%size] [1] : tensor<256xf32> to tensor<?xf32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]}
      ins(%slice : tensor<?xf32>) outs(%acc : tensor<i32>) attrs = {lowering_config = #config} {
      ^bb0(%in : f32, %out : i32):
        %idx = linalg.index 0 : index
        %idx_i32 = arith.index_cast %idx : index to i32
        %sum = arith.addi %idx_i32, %out : i32
        linalg.yield %sum : i32
    } -> tensor<i32>
    scf.yield %reduced : tensor<i32>
  }
  return %0 : tensor<i32>
}
// CHECK-LABEL: func.func @split_bounded_dynamic_with_linalg_index
// CHECK-NOT:     tensor.expand_shape
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK:             %[[INDEX:.+]] = linalg.index 0 : index
// CHECK:         scf.yield %[[GENERIC]]

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [0, 0, 16]>
func.func @split_bounded_dynamic_3d_reduction(%arg0 : tensor<4x5x128xi32>, %arg1 : tensor<4x5xi32>) -> tensor<4x5xi32> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %0 = scf.for %iv = %c0 to %c128 step %c64 iter_args(%acc = %arg1) -> (tensor<4x5xi32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 128, 64)>(%iv)
    %slice = tensor.extract_slice %arg0[0, 0, %iv] [4, 5, %size] [1, 1, 1] : tensor<4x5x128xi32> to tensor<4x5x?xi32>
    %reduced = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%slice : tensor<4x5x?xi32>) outs(%acc : tensor<4x5xi32>) attrs = {lowering_config = #config} {
      ^bb0(%in : i32, %out : i32):
        %sum = arith.addi %in, %out : i32
        linalg.yield %sum : i32
    } -> tensor<4x5xi32>
    scf.yield %reduced : tensor<4x5xi32>
  }
  return %0 : tensor<4x5xi32>
}
// CHECK-LABEL: func.func @split_bounded_dynamic_3d_reduction
// CHECK:         %[[SIZE:.+]] = affine.min
// CHECK:         %[[OUTER:.+]] = affine.apply
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape {{.*}} output_shape [1, 1, %[[OUTER]], 16]
// CHECK:         %[[INIT:.+]] = linalg.fill
// CHECK:         scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<1x1x16xi32>)
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK:         linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction"]

// -----

#config = #iree_cpu.lowering_config<vector_reduction = [16]>
func.func @split_dynamic_reduction_with_assume_int(%arg0 : tensor<?xi32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
  %assumed = util.assume.int %dim<udiv = 16> : index
  %slice = tensor.extract_slice %arg0[0] [%assumed] [1] : tensor<?xi32> to tensor<?xi32>
  %reduced = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
    ins(%slice : tensor<?xi32>) outs(%arg1 : tensor<i32>) attrs = {lowering_config = #config} {
    ^bb0(%in : i32, %out : i32):
      %sum = arith.addi %in, %out : i32
      linalg.yield %sum : i32
  } -> tensor<i32>
  return %reduced : tensor<i32>
}
// CHECK-LABEL: func.func @split_dynamic_reduction_with_assume_int
// CHECK:         util.assume.int
// CHECK:         %[[OUTER:.+]] = affine.apply
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape {{.*}} output_shape [%[[OUTER]], 16]
// CHECK:         %[[INIT:.+]] = linalg.fill
// CHECK:         scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<16xi32>)
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["reduction", "parallel"]
// CHECK:         linalg.generic
// CHECK-SAME:        iterator_types = ["reduction"]
