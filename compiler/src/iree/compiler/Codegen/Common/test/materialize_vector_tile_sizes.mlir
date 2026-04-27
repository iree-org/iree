// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-materialize-vector-tile-sizes))' --split-input-file %s | FileCheck %s

// Elementwise chain from to_layout anchor.

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1], batch_tile = [8], outer_tile = [1],
  thread_tile = [1], element_tile = [8],
  subgroup_strides = [0], thread_strides = [0]>

// CHECK-LABEL: @elementwise_from_anchor
func.func @elementwise_from_anchor(%arg0: tensor<63xf16>) -> tensor<63xf16> {
  %empty = tensor.empty() : tensor<63xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 64>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 64>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 64>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 64, 8>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 64>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 64>
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
// The fill-like %init generic is duplicatable, so it should NOT receive
// tile sizes.

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [8], batch_tile = [1], outer_tile = [1],
  thread_tile = [64], element_tile = [1],
  subgroup_strides = [1], thread_strides = [1]>

// CHECK-LABEL: @scf_for_propagation
func.func @scf_for_propagation(%arg0: tensor<512xf32>, %lb: index, %ub: index, %step: index) -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %cst = arith.constant 0.0 : f32
  // CHECK: linalg.generic
  // CHECK-NOT: iree_codegen.vector_tile_sizes
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
    // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 512>
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
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 64, 512>
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
  // CHECK-NOT: iree_codegen.vector_tile_sizes
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<512xf32>) -> tensor<512xf32>
  %if_result = scf.if %cond -> tensor<512xf32> {
    %laid_out = iree_vector_ext.to_layout %arg0 to layout(#layout_if) : tensor<512xf32>
    scf.yield %laid_out : tensor<512xf32>
  } else {
    scf.yield %fill : tensor<512xf32>
  }
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 512>
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

// -----

// Duplicatable linalg ops are specialized per use at materialization time.
// One linalg.fill feeding two consumers with different logical vector tile
// sizes should be duplicated and each clone should keep the tile size of its
// corresponding use.

#layout_fill_consumer_32 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 2], outer_tile = [1, 1],
  thread_tile = [16, 4], element_tile = [1, 4],
  subgroup_strides = [0, 0], thread_strides = [1, 16]>

#layout_fill_consumer_24 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 3], outer_tile = [1, 1],
  thread_tile = [16, 2], element_tile = [1, 4],
  subgroup_strides = [0, 0], thread_strides = [1, 16]>

// CHECK-LABEL: @result_and_fill_specialization
func.func @result_and_fill_specialization() -> (tensor<16x24xf32>, tensor<16x24xf32>) {
  %empty = tensor.empty() : tensor<16x24xf32>
  %zero = arith.constant 0.0 : f32
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x24xf32>
  // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[FILL24:.+]] = linalg.fill
  // CHECK-DAG: iree_codegen.vector_tile_sizes = array<i64: 16, 24>
  // CHECK-DAG: ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<16x24xf32>) -> tensor<16x24xf32>
  // CHECK-DAG: %[[FILL32:.+]] = linalg.fill
  // CHECK-DAG: iree_codegen.vector_tile_sizes = array<i64: 16, 32>
  // CHECK-DAG: ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<16x24xf32>) -> tensor<16x24xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<16x24xf32>) -> tensor<16x24xf32>
  // CHECK-DAG: %[[LAYOUT32:.+]] = iree_vector_ext.to_layout %[[FILL32]] to layout(#{{.+}}) : tensor<16x24xf32>
  // CHECK-DAG: %[[LAYOUT24:.+]] = iree_vector_ext.to_layout %[[FILL24]] to layout(#{{.+}}) : tensor<16x24xf32>
  // CHECK: return %[[LAYOUT32]], %[[LAYOUT24]] : tensor<16x24xf32>, tensor<16x24xf32>
  %laid_out_32 = iree_vector_ext.to_layout %fill to layout(#layout_fill_consumer_32) : tensor<16x24xf32>
  %laid_out_24 = iree_vector_ext.to_layout %fill to layout(#layout_fill_consumer_24) : tensor<16x24xf32>
  return %laid_out_32, %laid_out_24 : tensor<16x24xf32>, tensor<16x24xf32>
}

// -----

#layout_pack = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 32], outer_tile = [1, 1],
  thread_tile = [1, 1], element_tile = [8, 8],
  subgroup_strides = [0, 0], thread_strides = [0, 0]>

// CHECK-LABEL: @pack_forward_propagation
func.func @pack_forward_propagation(%arg0: tensor<8x256xf16>) -> tensor<8x8x32xf16> {
  %laid_out = iree_vector_ext.to_layout %arg0 to layout(#layout_pack) : tensor<8x256xf16>
  %empty_gen = tensor.empty() : tensor<8x256xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 256>
  %gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%laid_out : tensor<8x256xf16>) outs(%empty_gen : tensor<8x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<8x256xf16>
  %empty_pack = tensor.empty() : tensor<8x8x32xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %pack = linalg.pack %gen inner_dims_pos = [1] inner_tiles = [32]
    into %empty_pack : tensor<8x256xf16> -> tensor<8x8x32xf16>
  return %pack : tensor<8x8x32xf16>
}

// -----

#layout_pack_perm = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 32], outer_tile = [1, 1, 1],
  thread_tile = [1, 1, 1], element_tile = [4, 16, 8],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 0, 0]>

// CHECK-LABEL: @pack_forward_propagation_outer_perm
func.func @pack_forward_propagation_outer_perm(%arg0: tensor<4x16x256xf16>) -> tensor<8x4x16x32xf16> {
  %laid_out = iree_vector_ext.to_layout %arg0 to layout(#layout_pack_perm) : tensor<4x16x256xf16>
  %empty_gen = tensor.empty() : tensor<4x16x256xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 4, 16, 256>
  %gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%laid_out : tensor<4x16x256xf16>) outs(%empty_gen : tensor<4x16x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<4x16x256xf16>
  %empty_pack = tensor.empty() : tensor<8x4x16x32xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 4, 16, 32>
  %pack = linalg.pack %gen outer_dims_perm = [2, 0, 1] inner_dims_pos = [2] inner_tiles = [32]
    into %empty_pack : tensor<4x16x256xf16> -> tensor<8x4x16x32xf16>
  return %pack : tensor<8x4x16x32xf16>
}

// -----

// Pack with padding value: backward propagation through the pack should be
// blocked, so the generic producing the pack source should not get tile sizes.

#layout_pack_pad = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 1], outer_tile = [1, 1, 1],
  thread_tile = [1, 1, 1], element_tile = [8, 8, 32],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 0, 0]>

// CHECK-LABEL: @pack_no_backward_propagation_with_padding
func.func @pack_no_backward_propagation_with_padding(
    %arg0: tensor<8x256xf16>) -> tensor<8x8x32xf16> {
  %cst = arith.constant 0.0 : f16
  %empty_gen = tensor.empty() : tensor<8x256xf16>
  // CHECK: linalg.generic
  // CHECK-NOT: iree_codegen.vector_tile_sizes
  %gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<8x256xf16>) outs(%empty_gen : tensor<8x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<8x256xf16>
  %empty_pack = tensor.empty() : tensor<8x8x32xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %pack = linalg.pack %gen padding_value(%cst : f16)
    inner_dims_pos = [1] inner_tiles = [32]
    into %empty_pack : tensor<8x256xf16> -> tensor<8x8x32xf16>
  %result = iree_vector_ext.to_layout %pack to layout(#layout_pack_pad) : tensor<8x8x32xf16>
  return %result : tensor<8x8x32xf16>
}

// -----

#layout_unpack = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 32], outer_tile = [1, 1],
  thread_tile = [1, 1], element_tile = [8, 8],
  subgroup_strides = [0, 0], thread_strides = [0, 0]>

// CHECK-LABEL: @unpack_backward_propagation
func.func @unpack_backward_propagation(%arg0: tensor<8x8x32xf16>) -> tensor<8x256xf16> {
  %empty_unpack = tensor.empty() : tensor<8x256xf16>
  // CHECK: linalg.unpack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [1] inner_tiles = [32]
    into %empty_unpack : tensor<8x8x32xf16> -> tensor<8x256xf16>
  %empty_gen = tensor.empty() : tensor<8x256xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 256>
  %gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%unpack : tensor<8x256xf16>) outs(%empty_gen : tensor<8x256xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<8x256xf16>
  %result = iree_vector_ext.to_layout %gen to layout(#layout_unpack) : tensor<8x256xf16>
  return %result : tensor<8x256xf16>
}

// -----

#layout_chain = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1], batch_tile = [1, 32], outer_tile = [1, 1],
  thread_tile = [1, 1], element_tile = [8, 8],
  subgroup_strides = [0, 0], thread_strides = [0, 0]>

// CHECK-LABEL: @pack_unpack_chain
func.func @pack_unpack_chain(%arg0: tensor<8x256xf16>) -> tensor<8x256xf16> {
  %laid_out = iree_vector_ext.to_layout %arg0 to layout(#layout_chain) : tensor<8x256xf16>
  %empty_pack = tensor.empty() : tensor<8x8x32xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %pack = linalg.pack %laid_out inner_dims_pos = [1] inner_tiles = [32]
    into %empty_pack : tensor<8x256xf16> -> tensor<8x8x32xf16>
  %empty_gen = tensor.empty() : tensor<8x8x32xf16>
  // CHECK: linalg.generic
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %gen = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%pack : tensor<8x8x32xf16>) outs(%empty_gen : tensor<8x8x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %neg = arith.negf %in : f16
    linalg.yield %neg : f16
  } -> tensor<8x8x32xf16>
  %empty_unpack = tensor.empty() : tensor<8x256xf16>
  // CHECK: linalg.unpack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 8, 8, 32>
  %unpack = linalg.unpack %gen inner_dims_pos = [1] inner_tiles = [32]
    into %empty_unpack : tensor<8x8x32xf16> -> tensor<8x256xf16>
  return %unpack : tensor<8x256xf16>
}

// -----

#layout_pv_lhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 16, 4], element_tile = [1, 1, 4],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 1, 16]>

#layout_pv_rhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 4, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16], element_tile = [1, 4, 1],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 16, 1]>

#layout_pv_acc = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16], element_tile = [1, 4, 1],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 16, 1]>

// CHECK-LABEL: @inner_tiled_attention_pv
func.func @inner_tiled_attention_pv(
    %lhs: tensor<1x16x64xf16>,
    %rhs: tensor<1x64x64xf16>,
    %acc: tensor<1x16x64xf32>) -> tensor<1x16x64xf32> {
  %lhs_l = iree_vector_ext.to_layout %lhs to layout(#layout_pv_lhs) : tensor<1x16x64xf16>
  %rhs_l = iree_vector_ext.to_layout %rhs to layout(#layout_pv_rhs) : tensor<1x64x64xf16>
  %acc_l = iree_vector_ext.to_layout %acc to layout(#layout_pv_acc) : tensor<1x16x64xf32>
  %empty_lhs = tensor.empty() : tensor<1x1x4x16x16xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %pack_lhs = linalg.pack %lhs_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_lhs : tensor<1x16x64xf16> -> tensor<1x1x4x16x16xf16>
  %empty_rhs = tensor.empty() : tensor<1x4x4x16x16xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 4, 4, 16, 16>
  %pack_rhs = linalg.pack %rhs_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_rhs : tensor<1x64x64xf16> -> tensor<1x4x4x16x16xf16>
  %empty_acc = tensor.empty() : tensor<1x1x4x16x16xf32>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %pack_acc = linalg.pack %acc_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_acc : tensor<1x16x64xf32> -> tensor<1x1x4x16x16xf32>
  // CHECK: iree_codegen.inner_tiled
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 4>
  %result_packed = iree_codegen.inner_tiled ins(%pack_lhs, %pack_rhs) outs(%pack_acc) {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>,
                      #linalg.iterator_type<parallel>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<1x1x4x16x16xf16>, tensor<1x4x4x16x16xf16> into tensor<1x1x4x16x16xf32>
  %empty_result = tensor.empty() : tensor<1x16x64xf32>
  // CHECK: linalg.unpack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %result = linalg.unpack %result_packed inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_result : tensor<1x1x4x16x16xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

#layout_pv_lhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 16, 4], element_tile = [1, 1, 4],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 1, 16]>

#layout_pv_rhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 4, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16], element_tile = [1, 4, 1],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 16, 1]>

#layout_pv_acc = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1], batch_tile = [1, 1, 4], outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16], element_tile = [1, 4, 1],
  subgroup_strides = [0, 0, 0], thread_strides = [0, 16, 1]>

// CHECK-LABEL: @inner_tiled_dynamic
func.func @inner_tiled_dynamic(
    %lhs: tensor<1x?x?xf16>,
    %rhs: tensor<1x?x?xf16>,
    %acc: tensor<1x?x?xf32>) -> tensor<1x?x?xf32> {
  %lhs_l = iree_vector_ext.to_layout %lhs to layout(#layout_pv_lhs) : tensor<1x?x?xf16>
  %rhs_l = iree_vector_ext.to_layout %rhs to layout(#layout_pv_rhs) : tensor<1x?x?xf16>
  %acc_l = iree_vector_ext.to_layout %acc to layout(#layout_pv_acc) : tensor<1x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %lhs, %c0 : tensor<1x?x?xf16>
  %n = tensor.dim %rhs, %c0 : tensor<1x?x?xf16>
  %k = tensor.dim %lhs, %c1 : tensor<1x?x?xf16>
  %empty_lhs = tensor.empty(%m, %k) : tensor<1x?x?x16x16xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %pack_lhs = linalg.pack %lhs_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_lhs : tensor<1x?x?xf16> -> tensor<1x?x?x16x16xf16>
  %empty_rhs = tensor.empty(%n, %k) : tensor<1x?x?x16x16xf16>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 4, 4, 16, 16>
  %pack_rhs = linalg.pack %rhs_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_rhs : tensor<1x?x?xf16> -> tensor<1x?x?x16x16xf16>
  %empty_acc = tensor.empty(%m, %n) : tensor<1x?x?x16x16xf32>
  // CHECK: linalg.pack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %pack_acc = linalg.pack %acc_l inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_acc : tensor<1x?x?xf32> -> tensor<1x?x?x16x16xf32>
  // CHECK: iree_codegen.inner_tiled
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 4>
  %result_packed = iree_codegen.inner_tiled ins(%pack_lhs, %pack_rhs) outs(%pack_acc) {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>,
                      #linalg.iterator_type<parallel>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<1x?x?x16x16xf16>, tensor<1x?x?x16x16xf16> into tensor<1x?x?x16x16xf32>
  %empty_result = tensor.empty(%m, %n) : tensor<1x?x?xf32>
  // CHECK: linalg.unpack
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4, 16, 16>
  %result = linalg.unpack %result_packed inner_dims_pos = [1, 2] inner_tiles = [16, 16]
    into %empty_result : tensor<1x?x?x16x16xf32> -> tensor<1x?x?xf32>
  return %result : tensor<1x?x?xf32>
}

// -----

// Im2col: basic NHWC vectorization along K (output dim 2). K tile size (4)
// divides the innermost input dim C (640), so the analysis picks dim 2 as
// the vectorized dim with full size 4, and tiles the batch and M dims to 1.

#im2col_map_k = affine_map<(d0) -> (d0 * 4)>
// CHECK-LABEL: @im2col_tile_sizes_nhwc
func.func @im2col_tile_sizes_nhwc(
    %input: tensor<2x34x34x640xf32>, %m_off: index, %k: index
) -> tensor<2x2x4xf32> {
  %0 = tensor.empty() : tensor<2x2x4xf32>
  %k_off = affine.apply #im2col_map_k(%k)
  // CHECK: iree_linalg_ext.im2col
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 4>
  %1 = iree_linalg_ext.im2col
          strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
          offsets = [0, %m_off, %k_off] output_sizes = [[2], [32, 32], [3, 3, 640]]
          batch_pos = [0] m_pos = [1, 2] k_pos = [3]
          input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
          ins(%input : tensor<2x34x34x640xf32>)
          outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
  return %1 : tensor<2x2x4xf32>
}

// -----

// Im2col: non-vectorizable. input_k_perm = [1, 0] makes the innermost K
// non-contiguous in the input tensor, so no dimension can be vectorized
// with a contiguous slice. The analysis should not stamp a tile sizes
// attribute.

// CHECK-LABEL: @im2col_tile_sizes_non_contiguous
func.func @im2col_tile_sizes_non_contiguous(
    %input: tensor<1x3x2xf32>
) -> tensor<1x2x4xf32> {
  %0 = tensor.empty() : tensor<1x2x4xf32>
  // CHECK: iree_linalg_ext.im2col
  // CHECK-NOT: iree_codegen.vector_tile_sizes
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1] kernel_size = [2]
                          offsets = [0, 0, 0] output_sizes = [[1], [2], [2, 2]]
                          batch_pos = [0] m_pos = [1] k_pos = [2]
                          input_k_perm = [1, 0] output_perm = [0, 1, 2]
                          ins(%input : tensor<1x3x2xf32>)
                          outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}

// -----

// Im2col: wider vectorization. Vectorizes along the innermost channel dim
// with width 8. Non-vectorized spatial dims are tiled to 1.

// CHECK-LABEL: @im2col_tile_sizes_channel_width_8
func.func @im2col_tile_sizes_channel_width_8(
    %input: tensor<59x91x16x56xbf16>, %output: tensor<1x1x1x8xbf16>,
    %off0: index
) -> tensor<1x1x1x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %c5 = arith.constant 5 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : index
  // CHECK: iree_linalg_ext.im2col
  // CHECK-SAME: iree_codegen.vector_tile_sizes = array<i64: 1, 1, 1, 8>
  %result = iree_linalg_ext.im2col
      strides = [1, 1] dilations = [1, 1] kernel_size = [59, 91]
      offsets = [%off0, %c3, %c5, %c100]
      output_sizes = [[64], [16], [3, 3], [59, 91]]
      batch_pos = [3, 2] m_pos = [0, 1] k_pos = []
      input_k_perm = [0, 1] output_perm = [2, 3, 1, 0]
      input_pad_low = [1, 1, 0, 0] input_pad_high = [1, 1, 0, 8]
      pad_value(%cst : bf16)
      ins(%input : tensor<59x91x16x56xbf16>)
      outs(%output : tensor<1x1x1x8xbf16>) -> tensor<1x1x1x8xbf16>
  return %result : tensor<1x1x1x8xbf16>
}
