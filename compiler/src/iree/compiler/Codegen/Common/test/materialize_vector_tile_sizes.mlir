// RUN: iree-opt --pass-pipeline='builtin.module(any(iree-codegen-materialize-vector-tile-sizes))' --split-input-file %s | FileCheck %s

// Elementwise chain from to_layout anchor.

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1], batch_tile = [8], outer_tile = [1],
  thread_tile = [1], element_tile = [8],
  subgroup_strides = [0], thread_strides = [0]>

// CHECK-LABEL: @elementwise_from_anchor
func.func @elementwise_from_anchor(%arg0: tensor<63xf16>) -> tensor<63xf16> {
  %empty = tensor.empty() : tensor<63xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 64>]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<63xf16>) outs(%empty : tensor<63xf16>) {
  ^bb0(%in: f16, %out: f16):
    %add = arith.addf %in, %in : f16
    linalg.yield %add : f16
  } -> tensor<63xf16>
  %1 = iree_vector_ext.to_layout %0 to layout(#layout) : tensor<63xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 64>]
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%1 : tensor<63xf16>) outs(%empty : tensor<63xf16>) {
  ^bb0(%in: f16, %out: f16):
    %mul = arith.mulf %in, %in : f16
    linalg.yield %mul : f16
  } -> tensor<63xf16>
  return %2 : tensor<63xf16>
}

// -----

// Chain propagation with transpose: tile sizes must propagate through
// generic A's result to generic B, with B using a transposed indexing map.

#layout_2d = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 8], outer_tile = [1, 1],
  thread_tile = [1, 1], element_tile = [8, 8],
  subgroup_strides = [0, 0], thread_strides = [0, 0]>

// CHECK-LABEL: @chain_propagation_transpose
func.func @chain_propagation_transpose(
    %arg0: tensor<8x64xf32>, %arg1: tensor<8x64xf32>) -> tensor<64x8xf32> {
  %a = iree_vector_ext.to_layout %arg0 to layout(#layout_2d) : tensor<8x64xf32>
  %empty_ab = tensor.empty() : tensor<8x64xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 8>, array<i64: 64>]
  %ab = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %arg1 : tensor<8x64xf32>, tensor<8x64xf32>)
    outs(%empty_ab : tensor<8x64xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<8x64xf32>
  %empty_t = tensor.empty() : tensor<64x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 64>, array<i64: 8>]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%ab : tensor<8x64xf32>) outs(%empty_t : tensor<64x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<64x8xf32>
  return %result : tensor<64x8xf32>
}

// -----

// Chain propagation with dynamic shapes: tile sizes propagate the same way
// regardless of whether tensor dimensions are static or dynamic.

#layout_dyn = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 8], outer_tile = [1, 1],
  thread_tile = [1, 1], element_tile = [8, 8],
  subgroup_strides = [0, 0], thread_strides = [0, 0]>

// CHECK-LABEL: @chain_propagation_dynamic
func.func @chain_propagation_dynamic(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %a = iree_vector_ext.to_layout %arg0 to layout(#layout_dyn) : tensor<?x?xf32>
  %empty_ab = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 8>, array<i64: 64>]
  %ab = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%empty_ab : tensor<?x?xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>
  %empty_c = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 8>, array<i64: 64>]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%ab : tensor<?x?xf32>) outs(%empty_c : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

// scf.for propagation through iter_args.
// The to_layout inside the loop should propagate tile sizes to the
// loop iter_args and through the scf.yield.

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [8], batch_tile = [1], outer_tile = [1],
  thread_tile = [64], element_tile = [1],
  subgroup_strides = [1], thread_strides = [1]>

// CHECK-LABEL: @scf_for_propagation
func.func @scf_for_propagation(%arg0: tensor<512xf32>, %lb: index, %ub: index, %step: index) -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %cst = arith.constant 0.0 : f32
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 512>]
  %init = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%cst : f32) outs(%empty : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<512xf32>
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %init) -> tensor<512xf32> {
    %laid_out = iree_vector_ext.to_layout %iter to layout(#layout) : tensor<512xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 512>]
    %updated = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%laid_out, %arg0 : tensor<512xf32>, tensor<512xf32>) outs(%empty : tensor<512xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %add = arith.addf %in0, %in1 : f32
      linalg.yield %add : f32
    } -> tensor<512xf32>
    scf.yield %updated : tensor<512xf32>
  }
  return %result : tensor<512xf32>
}

// -----

#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1], batch_tile = [8], outer_tile = [1],
  thread_tile = [1], element_tile = [8],
  subgroup_strides = [0], thread_strides = [0]>

#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [8, 1], batch_tile = [1, 8], outer_tile = [1, 1],
  thread_tile = [64, 1], element_tile = [1, 8],
  subgroup_strides = [1, 0], thread_strides = [1, 0]>

#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [8], batch_tile = [1], outer_tile = [1],
  thread_tile = [64], element_tile = [1],
  subgroup_strides = [1], thread_strides = [1]>

// CHECK-LABEL: @contraction_indexing_maps
func.func @contraction_indexing_maps(
    %a: tensor<63xf16>, %b: tensor<512x63xf16>, %c: tensor<512xf32>) -> tensor<512xf32> {
  %al = iree_vector_ext.to_layout %a to layout(#layout_a) : tensor<63xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layout_b) : tensor<512x63xf16>
  %cl = iree_vector_ext.to_layout %c to layout(#layout_c) : tensor<512xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 64>, array<i64: 512>]
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d1)>
    ],
    iterator_types = ["reduction", "parallel"]
  } ins(%al, %bl : tensor<63xf16>, tensor<512x63xf16>) outs(%cl : tensor<512xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %ext0 = arith.extf %in0 : f16 to f32
    %ext1 = arith.extf %in1 : f16 to f32
    %mul = arith.mulf %ext0, %ext1 : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// -----

// scf.if propagation: tile size from the to_layout inside one branch
// should propagate through the scf.if result to consumers outside.

#layout_if = #iree_vector_ext.nested_layout<
  subgroup_tile = [8], batch_tile = [1], outer_tile = [1],
  thread_tile = [64], element_tile = [1],
  subgroup_strides = [1], thread_strides = [1]>

// CHECK-LABEL: @scf_if_propagation
func.func @scf_if_propagation(%arg0: tensor<512xf32>, %cond: i1) -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<512xf32>) -> tensor<512xf32>
  %if_result = scf.if %cond -> tensor<512xf32> {
    %laid_out = iree_vector_ext.to_layout %arg0 to layout(#layout_if) : tensor<512xf32>
    scf.yield %laid_out : tensor<512xf32>
  } else {
    scf.yield %fill : tensor<512xf32>
  }
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = [array<i64: 512>]
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%if_result : tensor<512xf32>) outs(%empty : tensor<512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  } -> tensor<512xf32>
  return %result : tensor<512xf32>
}
