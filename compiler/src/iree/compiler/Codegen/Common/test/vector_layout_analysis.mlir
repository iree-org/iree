// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>, %cond: i1) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %c = arith.mulf %rootl, %b : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %e = arith.select %cond, %c, %d : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    func.return %e : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Enforce the layout from the transfer_write to everyone
builtin.module attributes { transform.with_named_sequence } {
  func.func @enforce_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0 = arith.constant dense<0.0> : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %c = arith.mulf %cst0, %b : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %dl = iree_vector_ext.to_layout %d to layout(#layout) : vector<16x16xf16>
    vector.transfer_write %dl, %arr[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// First propagate the layout, and then enforce it up.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_and_enforce(%arr: memref<16x16xf16>, %arr2: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %root2 = vector.transfer_read %arr2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %c = arith.mulf %rootl, %b : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %e = arith.divf %d, %root2 : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    func.return %e : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate and enforce through reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 8]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    %root_red = vector.multi_reduction<add>, %rootl, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{element_tile = [8]}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate and enforce through transpose and then reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_and_reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 8]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %root_transpose = vector.transpose %rootl, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    // expected-remark @above {{element_tile = [8, 16]}}
    %root_red = vector.multi_reduction<add>, %root_transpose, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layoutA = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [32, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layoutB = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [128, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layoutC = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [128, 32],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

// Propagate through vector.contract.
builtin.module attributes { transform.with_named_sequence } {
  func.func @contract(%A : vector<32x64xf16>, %B : vector<128x64xf16>, %C : vector<128x32xf32>) -> vector<128x32xf32> {
    %a = iree_vector_ext.to_layout %A to layout(#layoutA) : vector<32x64xf16>
    %b = iree_vector_ext.to_layout %B to layout(#layoutB) : vector<128x64xf16>
    %c = iree_vector_ext.to_layout %C to layout(#layoutC) : vector<128x32xf32>

    // Check if the layout of %C was properly propagated to %D.
    // expected-remark @below {{element_tile = [128, 32]}}
    %D = vector.contract
        {indexing_maps = [#map1, #map2, #map3],
         iterator_types = ["parallel", "parallel", "reduction"],
         kind = #vector.kind<add>
        } %b, %a, %c : vector<128x64xf16>, vector<32x64xf16> into vector<128x32xf32>

    func.return %D : vector<128x32xf32>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @gather(%base: memref<16x16xf16>, %arr: memref<16x16xindex>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %mask = arith.constant dense<true> : vector<16x16xi1>
    // expected-remark @above {{element_tile = [16, 16]}}
    %pass = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    %index = vector.transfer_read %arr[%c0, %c0], %c0 {in_bounds = [true, true]} : memref<16x16xindex>, vector<16x16xindex>
    // expected-remark @above {{element_tile = [16, 16]}}
    %index_dist = iree_vector_ext.to_layout %index to layout(#layout) : vector<16x16xindex>
    %c = vector.gather %base[%c0, %c0] [%index_dist], %mask, %pass : memref<16x16xf16>, vector<16x16xindex>, vector<16x16xi1>, vector<16x16xf16> into vector<16x16xf16>
    // expected-remark @above {{element_tile = [16, 16]}}
    func.return %c : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// This test checks that we can resolve layouts through arith.select
// properly and that our layout analysis is not emitting redundant
// to_layout conversions in between anchor ops.

// Useful proxy for ensuring that layout conversions on attention
// happens where we intend it to happen.

#layoutA = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 4],
  thread_tile = [32, 2],
  element_tile = [1, 4],

  subgroup_strides = [0, 0],
  thread_strides   = [2, 1]
>

#layoutB = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [32, 4],
  element_tile = [1, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [4, 1]
>

builtin.module attributes { transform.with_named_sequence } {
  func.func @resolve_select(%A : vector<64x64xf16>, %B : vector<64x64xf16>, %condition : i1) -> vector<64x64xf16> {
    %a = iree_vector_ext.to_layout %A to layout(#layoutA) : vector<64x64xf16>
    %b = iree_vector_ext.to_layout %B to layout(#layoutB) : vector<64x64xf16>
    // expected-remark @below {{element_tile = [1, 4]}}
    %offset_0 = arith.constant dense<2.0> : vector<64x64xf16>
    // expected-remark @below {{element_tile = [1, 4]}}
    %offset_1 = arith.constant dense<4.0> : vector<64x64xf16>

    // expected-remark @below {{element_tile = [1, 4]}}
    %sel = arith.select %condition, %offset_0, %offset_1 : vector<64x64xf16>
    // expected-remark @below {{element_tile = [1, 4]}}
    %add = arith.addf %a, %sel : vector<64x64xf16>
    %add_layout = iree_vector_ext.to_layout %add to layout(#layoutB) : vector<64x64xf16>
    // CHECK-COUNT-3: iree_vector_ext.to_layout
    // expected-remark @below {{element_tile = [1, 8]}}
    %add_1 = arith.addf %add_layout, %b : vector<64x64xf16>
    func.return %add_1 : vector<64x64xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate and enforce through scf.for
builtin.module attributes { transform.with_named_sequence } {
  func.func @scffor(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{element_tile = [16]}}
    %out = scf.for %iv = %c0 to %c1024 step %c1 iter_args(%arg1 = %cst0_1) -> (vector<16xf16>) {
      // expected-remark @above {{element_tile = [16]}}
      %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
      // expected-remark @above {{element_tile = [16, 8]}}
      %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
      %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
      // expected-remark @above {{element_tile = [16]}}
      %root_transpose = vector.transpose %rootl, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
      // expected-remark @above {{element_tile = [8, 16]}}
      %root_red = vector.multi_reduction<add>, %root_transpose, %arg1 [0]  : vector<16x16xf16> to vector<16xf16>
      // expected-remark @above {{element_tile = [16]}}
      %c = arith.mulf %root_red, %b : vector<16xf16>
      // expected-remark @above {{element_tile = [16]}}
      %d = arith.addf %c, %a : vector<16xf16>
      // expected-remark @above {{element_tile = [16]}}
      %e = arith.divf %d, %root2 : vector<16xf16>
      // expected-remark @above {{element_tile = [16]}}
      scf.yield %e : vector<16xf16>
    }

    func.return %out : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 4]
>

// Propagate and enforce through reduction along dim 0.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{thread_strides = [4]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{thread_strides = [1, 4]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %root_red = vector.multi_reduction<add>, %rootl, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{thread_strides = [4]}}
    %c = arith.mulf %root_red, %a : vector<16xf16>
    // expected-remark @above {{thread_strides = [4]}}
    func.return %c : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 4]
>

// Propagate and enforce through reduction along dim 1.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{thread_strides = [1]}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{thread_strides = [1, 4]}}
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
    %root_red = vector.multi_reduction<add>, %rootl, %cst0_1 [1]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{thread_strides = [1]}}
    %c = arith.mulf %root_red, %a : vector<16xf16>
    // expected-remark @above {{thread_strides = [1]}}
    func.return %c : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1, 1],
  batch_tile = [1, 2, 4],
  outer_tile = [1, 1, 1],
  thread_tile = [4, 8, 2],
  element_tile = [4, 1, 2],

  subgroup_strides = [1, 2, 2],
  thread_strides   = [1, 4, 32]
>

// Propagate and enforce through reduction along dim 1.
// The printing of this layout is too long for a remark. Just verify that
// the subgroup/thread bases are what we expect.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_3d(%arr: memref<32x32x32xf16>) -> () {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark-re @below {{thread_strides = [1, 4, 32]}}
    %root = vector.transfer_read %arr[%c0, %c0, %c0], %cst_0 {
      in_bounds = [true, true, true]
    } : memref<32x32x32xf16>, vector<32x16x16xf16>
    %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<32x16x16xf16>
    %t = vector.transpose %rootl, [1, 2, 0] : vector<32x16x16xf16> to vector<16x16x32xf16>
    // expected-remark-re @above {{thread_strides = [4, 32, 1]}}
    vector.transfer_write %t, %arr[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x16x32xf16>, memref<32x32x32xf16>
    func.return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [4, 1],
  outer_tile = [1, 1],
  thread_tile = [32, 4],
  element_tile = [1, 32],

  subgroup_strides = [1, 1],
  thread_strides = [1, 32]
>

// Propagate and enforce layout through broadcast transpose and broadcast.
// Main thing we want to see here is the subgroup_strides and thread_strides
// are being determined properly.
builtin.module attributes { transform.with_named_sequence } {
  func.func @broadcast_transpose(%quant :  memref<128x128xi4>, %scale : memref<128xf16>, %arr: memref<128x128xf16>) -> () {
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i4 = arith.constant 0 : i4
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %quant[%c0, %c0], %c0_i4 {in_bounds = [true, true]} : memref<128x128xi4>, vector<128x128xi4>
    // expected-remark @above {{thread_strides = [1, 32]}}
    %00 = iree_vector_ext.to_layout %0 to layout(#layout) : vector<128x128xi4>
    %1 = vector.transfer_read %scale[%c0], %cst {in_bounds = [true]} : memref<128xf16>, vector<128xf16>
    // expected-remark @above {{thread_strides = [1]}}
    %2 = vector.broadcast %1 : vector<128xf16> to vector<128x128xf16>
    // expected-remark @above {{thread_strides = [32, 1]}}
    %3 = vector.transpose %2, [1, 0] : vector<128x128xf16> to vector<128x128xf16>
    // expected-remark @above {{thread_strides = [1, 32]}}
    %4 = arith.extui %00 : vector<128x128xi4> to vector<128x128xi32>
    // expected-remark @above {{thread_strides = [1, 32]}}
    %5 = arith.uitofp %4 : vector<128x128xi32> to vector<128x128xf16>
    // expected-remark @above {{thread_strides = [1, 32]}}
    %6 = arith.mulf %5, %3 : vector<128x128xf16>
    // expected-remark @above {{thread_strides = [1, 32]}}
    vector.transfer_write %6, %arr[%c0, %c0] {in_bounds = [true, true]} : vector<128x128xf16>, memref<128x128xf16>
    func.return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#contract_layout = #iree_vector_ext.nested_layout<
    subgroup_tile = [1, 1],
    batch_tile = [3, 2],
    outer_tile = [4, 1],
    thread_tile = [2, 32],
    element_tile = [4, 1],

    subgroup_strides = [0, 0],
    thread_strides = [32, 1]
>

// This test ensures that we are not running into ops not dominating constantOp operands after layout analysis.
// We simulate that by doing elmentwise op on the value with "layout" i.e scaled_lhs after scaled_rhs.
// If not handled properly, will generate constOp before "scaled_lhs", but would get used also by "scaled_rhs".
builtin.module attributes { transform.with_named_sequence } {
  func.func @handle_multiuse_constant(%lhs: vector<96x64xf16>, %rhs: vector<96x64xf16>, %arr: memref<96x64xf16>) -> () {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant dense<1.562500e-02> : vector<96x64xf16>
    // expected-remark @above {{thread_strides = [32, 1]}}
    %lhs_layout = iree_vector_ext.to_layout %lhs to layout(#contract_layout) : vector<96x64xf16>

    %scaled_rhs = arith.mulf %rhs, %cst_1 : vector<96x64xf16>
    // expected-remark @above {{thread_strides = [32, 1]}}
    %scaled_lhs = arith.mulf %lhs_layout, %cst_1 : vector<96x64xf16>
    // expected-remark @above {{thread_strides = [32, 1]}}
    %add = arith.addf %scaled_lhs, %scaled_rhs : vector<96x64xf16>
    // expected-remark @above {{thread_strides = [32, 1]}}
    vector.transfer_write %add, %arr[%c0, %c0] {in_bounds = [true, true]} : vector<96x64xf16>, memref<96x64xf16>
    func.return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layoutA = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [1],
  thread_tile   = [1],
  element_tile  = [64],

  subgroup_strides = [0],
  thread_strides   = [0]
>

#layoutB = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [64],
  outer_tile    = [1],
  thread_tile   = [1],
  element_tile  = [1],

  subgroup_strides = [0],
  thread_strides   = [0]
>

builtin.module attributes { transform.with_named_sequence } {
  func.func @handle_multiuse_step(%lhs: vector<64xindex>, %rhs: vector<64xindex>) -> (vector<64xindex>, vector<64xindex>) {
    %l_lhs = iree_vector_ext.to_layout %lhs to layout(#layoutA) : vector<64xindex>
    %r_lhs = iree_vector_ext.to_layout %rhs to layout(#layoutB) : vector<64xindex>
    %cst = vector.step : vector<64xindex>
    // expected-remark @above {{element_tile = [1]}}
    // expected-remark @above {{element_tile = [64]}}
    %scaled_lhs = arith.muli %cst, %lhs : vector<64xindex>
    // expected-remark @above {{element_tile = [64]}}
    %scaled_rhs = arith.muli %cst, %rhs : vector<64xindex>
    // expected-remark @above {{element_tile = [1]}}
    func.return %scaled_lhs, %scaled_rhs : vector<64xindex>, vector<64xindex>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1, 1],
  batch_tile = [1, 2, 4],
  outer_tile = [1, 1, 1],
  thread_tile = [4, 8, 2],
  element_tile = [4, 1, 2],

  subgroup_strides = [1, 2, 2],
  thread_strides   = [1, 4, 32]
>

/// Invalid anchor tests

// Rank mismatch anchor.
builtin.module attributes { transform.with_named_sequence } {
  func.func @invalid_rank_nested_layout_anchor(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c = arith.addf %a, %b : vector<16x16xf16>
    %cl = iree_vector_ext.to_layout %c to layout(#layout) : vector<16x16xf16>
    // expected-error @above {{Rank of vector (2) does not match rank of layout (3)}}
    func.return %cl : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout2 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 4],
  outer_tile = [1, 1],
  thread_tile = [8, 2],
  element_tile = [2, 2],

  subgroup_strides = [0, 0],
  thread_strides   = [1, 8]
>

// Size mismatch anchor.
builtin.module attributes { transform.with_named_sequence } {
  func.func @invalid_size_nested_layout_anchor(%a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c = arith.addf %a, %b : vector<16x16xf16>
    %cl = iree_vector_ext.to_layout %c to layout(#layout2) : vector<16x16xf16>
    // expected-error @above {{Vector shape: [16, 16] does not match the layout (nested_layout<subgroup_tile = [1, 1], batch_tile = [2, 4], outer_tile = [1, 1], thread_tile = [8, 2], element_tile = [2, 2], subgroup_strides = [0, 0], thread_strides = [1, 8]>) at dim 0. Dimension expected by layout: 32 actual: 16}}
    func.return %cl : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 8],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Propagate and enforce through scf.for
builtin.module attributes { transform.with_named_sequence } {
  func.func @scffor(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<f16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant dense<0.000000e+00> : vector<f16>
    %cst_0 = arith.constant 0.0 : f16

    %out = scf.for %iv = %c0 to %c1024 step %c1 iter_args(%arg1 = %cst) -> (vector<f16>) {
      %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
      // expected-remark @above {{element_tile = [16, 8]}}
      %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xf16>
      %init = vector.extractelement %arg1[] : vector<f16>
      %root_red = vector.multi_reduction<add>, %rootl, %init [0, 1]  : vector<16x16xf16> to f16
      %c = vector.broadcast %root_red : f16 to vector<f16>
      scf.yield %c : vector<f16>
    }

    func.return %out : vector<f16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}
