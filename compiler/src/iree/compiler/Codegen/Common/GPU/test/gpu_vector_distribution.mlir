// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile    = [2, 1, 1],
  batch_tile       = [8, 2, 4],
  outer_tile       = [1, 4, 4],
  thread_tile      = [8, 2, 4],
  element_tile     = [1, 8, 2],
  subgroup_strides = [1, 1, 1],
  thread_strides   = [1, 8, 16]
>

// CHECK-LABEL: @distribute_elementwise_nested_layout_f16
func.func @distribute_elementwise_nested_layout_f16(%a: vector<128x128x128xf16>, %b: vector<128x128x128xf16>) -> vector<128x128x128xf16> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<8x2x4x1x4x4x1x8x2xf16>
  %root = arith.constant dense<0.0> : vector<128x128x128xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#nested) : vector<128x128x128xf16>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<128x128x128xf16> -> vector<8x2x4x1x4x4x1x8x2xf16>
  // CHECK-DAG: %[[C:.*]] = arith.mulf %[[B]], %[[ROOT]] : vector<8x2x4x1x4x4x1x8x2xf16>
  %c = arith.mulf %rootl, %b : vector<128x128x128xf16>
  // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<128x128x128xf16> -> vector<8x2x4x1x4x4x1x8x2xf16>
  // CHECK-DAG: %[[D:.*]] = arith.addf %[[C]], %[[A]] fastmath<reassoc,nnan> : vector<8x2x4x1x4x4x1x8x2xf16>
  %d = arith.addf %c, %a fastmath<reassoc,nnan> : vector<128x128x128xf16>
  // CHECK: iree_vector_ext.to_simd %[[D]] : vector<8x2x4x1x4x4x1x8x2xf16> -> vector<128x128x128xf16>
  return %d : vector<128x128x128xf16>
}

// CHECK-LABEL: @distribute_poison
func.func @distribute_poison() -> vector<128x128x128xf16> {
  // CHECK: ub.poison : vector<8x2x4x1x4x4x1x8x2xf16>
  %root = ub.poison : vector<128x128x128xf16>
  %rootl = iree_vector_ext.to_layout %root to layout(#nested) : vector<128x128x128xf16>
  return %rootl: vector<128x128x128xf16>
}

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [1, 1],
  element_tile     = [16, 16],
  subgroup_strides = [1, 1],
  thread_strides   = [1, 1]
>

// CHECK-LABEL: @distribute_scf_for
func.func @distribute_scf_for(%a: vector<16x16xi32>, %b: vector<16x16xi32>) -> vector<16x16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<1x1x1x1x16x16xi32>
  %root = arith.constant dense<0> : vector<16x16xi32>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout) : vector<16x16xi32>
  // CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<1x1x1x1x16x16xi32>)
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %rootl) -> (vector<16x16xi32>) {
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<1x1x1x1x16x16xi32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ARG0]], %[[B]] : vector<1x1x1x1x16x16xi32>
    %c = arith.muli %arg0, %b : vector<16x16xi32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16x16xi32> -> vector<1x1x1x1x16x16xi32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<1x1x1x1x16x16xi32>
    %d = arith.addi %c, %a : vector<16x16xi32>
    // CHECK: scf.yield %[[D]] : vector<1x1x1x1x16x16xi32>
    scf.yield %d : vector<16x16xi32>
  }
  return %out : vector<16x16xi32>
}

#layout_0d = #iree_vector_ext.nested_layout<
  subgroup_tile    = [],
  batch_tile       = [],
  outer_tile       = [],
  thread_tile      = [],
  element_tile     = [],
  subgroup_strides = [],
  thread_strides   = []
>

// CHECK-LABEL: @distribute_scf_for_0d
func.func @distribute_scf_for_0d(%a: vector<i32>, %b: vector<i32>) -> vector<i32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_0 = arith.constant 0 : i32
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0> : vector<i32>
  %root = arith.constant dense<0> : vector<i32>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_0d) : vector<i32>
  // CHECK: iter_args(%[[ARG0:.*]] = %[[ROOT]]) -> (vector<i32>)
  %out = scf.for %i = %c0 to %c128 step %c1 iter_args(%arg0 = %rootl) -> (vector<i32>) {
    // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<i32> -> vector<i32>
    // CHECK-DAG: %[[C:.*]] = arith.muli %[[ARG0]], %[[B]] : vector<i32>
    %c = arith.muli %arg0, %b : vector<i32>
    // CHECK-DAG: %[[A:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<i32> -> vector<i32>
    // CHECK-DAG: %[[D:.*]] = arith.addi %[[C]], %[[A]] : vector<i32>
    %d = arith.addi %c, %a : vector<i32>
    // CHECK: scf.yield %[[D]] : vector<i32>
    scf.yield %d : vector<i32>
  }
  return %out : vector<i32>
}

// CHECK-LABEL: @distribute_scalar_extract
func.func @distribute_scalar_extract(%a: f16, %b: vector<f16>) -> f16 {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.0 : f16
  // CHECK: %[[ROOT:.*]] = arith.constant dense<0.000000e+00> : vector<f16>
  %root = arith.constant dense<0.0> : vector<f16>
  %rootl = iree_vector_ext.to_layout %root to layout(#layout_0d) : vector<f16>
  // CHECK-DAG: %[[B:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<f16> -> vector<f16>
  // CHECK-DAG: %[[C:.*]] = arith.mulf %[[B]], %[[ROOT]] : vector<f16>
  // CHECK-DAG: %[[SCALAR:.*]] = vector.extract %[[C]][] : f16 from vector<f16>
  %c = arith.mulf %rootl, %b : vector<f16>
  %scalar = vector.extract %c[] : f16 from vector<f16>
  // CHECK-DAG: %[[D:.*]] = arith.addf %[[SCALAR]], %{{.*}} : f16
  %d = arith.addf %scalar, %a : f16
  return %d : f16
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
