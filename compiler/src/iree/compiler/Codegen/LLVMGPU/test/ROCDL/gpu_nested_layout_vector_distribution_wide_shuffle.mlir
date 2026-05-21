// RUN: iree-opt --iree-transform-dialect-interpreter --iree-gpu-test-target=gfx942 --split-input-file --canonicalize -mlir-print-local-scope --cse %s | FileCheck %s

// AMDGPU-only nested-layout vector distribution cases. These exercise paths
// gated on `targetSupportsShuffleBitwidth` returning true for >32-bit element
// types, which today requires an AMD target with a decomposing `gpu.shuffle`
// lowering. Target-agnostic cases (<=32-bit elements) live in the per-op
// distribution test files (multi_reduce, argcompare, scan).

// f64 vector.multi_reduction on AMDGPU: passes the bitwidth gate, so the
// pattern fires and emits `gpu.subgroup_reduce ... : (f64) -> f64`.

#nested_f64 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @reduce_f64_1d_dim(%arg0: vector<32x32xf64>, %arg1: vector<32xf64>) -> vector<32xf64> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested_f64) : vector<32x32xf64>
  %0 = vector.multi_reduction <maximumf>, %arg0l, %arg1 [1] : vector<32x32xf64> to vector<32xf64>
  return %0 : vector<32xf64>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reduce_f64_1d_dim
// Local reduction on the distributed batch shape.
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<2x2x1x1x1x4xf64> to vector<2x1x1xf64>
// Cross-thread reduction on the 64-bit element type.
// CHECK: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 4, stride = 16) : (f64) -> f64

// -----

// i64 vector.multi_reduction on AMDGPU: parallel to @reduce_f64_1d_dim,
// exercises the same 64-bit shuffle gate for an integer reduction kind.

#nested_i64 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @reduce_i64_1d_dim(%arg0: vector<32x32xi64>, %arg1: vector<32xi64>) -> vector<32xi64> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested_i64) : vector<32x32xi64>
  %0 = vector.multi_reduction <add>, %arg0l, %arg1 [1] : vector<32x32xi64> to vector<32xi64>
  return %0 : vector<32xi64>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reduce_i64_1d_dim
// Local reduction on the distributed batch shape.
// CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<2x2x1x1x1x4xi64> to vector<2x1x1xi64>
// Cross-thread reduction on the 64-bit element type.
// CHECK: gpu.subgroup_reduce add %{{.*}} cluster(size = 4, stride = 16) : (i64) -> i64

// -----

// i128 vector.multi_reduction on AMDGPU: targetSupportsShuffleBitwidth has no
// upper bound on AMD, and the ROCDL gpu.shuffle lowering generically
// decomposes wide values into i32 ds_bpermute chunks. The pattern fires and
// emits `gpu.subgroup_reduce ... : (i128) -> i128`, which lowers to four
// 32-bit shuffles.

#nested_i128 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @reduce_i128_1d_dim(%arg0: vector<32x32xi128>, %arg1: vector<32xi128>) -> vector<32xi128> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested_i128) : vector<32x32xi128>
  %0 = vector.multi_reduction <add>, %arg0l, %arg1 [1] : vector<32x32xi128> to vector<32xi128>
  return %0 : vector<32xi128>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reduce_i128_1d_dim
// Local reduction on the distributed batch shape.
// CHECK: vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<2x2x1x1x1x4xi128> to vector<2x1x1xi128>
// Cross-thread reduction on the 128-bit element type.
// CHECK: gpu.subgroup_reduce add %{{.*}} cluster(size = 4, stride = 16) : (i128) -> i128

// -----

// i64 arg_compare on AMDGPU: passes the bitwidth gate. The commutative
// comparator selects the ballot path: subgroup_reduce on i64 + a single
// i32 shuffle for the winning lane index.

#layout_1d_thread_only = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [64],
  element_tile = [1],

  subgroup_strides = [1],
  thread_strides   = [1]
>

#layout_0d_thread_only = #iree_vector_ext.nested_layout<
  subgroup_tile = [],
  batch_tile = [],
  outer_tile = [],
  thread_tile = [],
  element_tile = [],

  subgroup_strides = [],
  thread_strides   = []
>

// CHECK-LABEL: func @argcompare_i64
// CHECK-NOT: iree_vector_ext.arg_compare
// CHECK: gpu.subgroup_reduce maxsi {{.*}} : (i64) -> i64
// CHECK: gpu.shuffle idx {{.*}} : i32
func.func @argcompare_i64(
    %input: vector<64xi64>,
    %input_idx: vector<64xi32>,
    %init_val: vector<i64>,
    %init_idx: vector<i32>) -> (vector<i64>, vector<i32>) {
  %v = iree_vector_ext.to_layout %input to layout(#layout_1d_thread_only) : vector<64xi64>
  %i = iree_vector_ext.to_layout %input_idx to layout(#layout_1d_thread_only) : vector<64xi32>
  %iv = iree_vector_ext.to_layout %init_val to layout(#layout_0d_thread_only) : vector<i64>
  %ii = iree_vector_ext.to_layout %init_idx to layout(#layout_0d_thread_only) : vector<i32>
  %res:2 = iree_vector_ext.arg_compare dimension(0)
      ins(%v, %i : vector<64xi64>, vector<64xi32>)
      inits(%iv, %ii : vector<i64>, vector<i32>) {
    ^bb0(%a: i64, %b: i64):
      %cmp = arith.cmpi sgt, %a, %b : i64
      iree_vector_ext.yield %cmp : i1
  } -> vector<i64>, vector<i32>
  return %res#0, %res#1 : vector<i64>, vector<i32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

// Carve-out: 64-bit scan on gfx942. Even though the AMDGPU target accepts
// >32-bit shuffles for reduction/arg_compare, scan is intentionally kept on
// the 32-bit ceiling. The pattern must NOT fire here: the original
// `vector.scan` survives and no `iree_gpu.subgroup_scan` is emitted.

#layout_scan_f64 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [4],

  subgroup_strides = [1],
  thread_strides   = [1]
>

// CHECK-LABEL: @scan_f64_gfx942_carve_out
func.func @scan_f64_gfx942_carve_out(%src: vector<16xf64>, %init: vector<f64>) -> (vector<16xf64>, vector<f64>) {
  %src_l = iree_vector_ext.to_layout %src to layout(#layout_scan_f64) : vector<16xf64>
  %out:2 = vector.scan <add>, %src_l, %init {inclusive = true, reduction_dim = 0 : i64}
    : vector<16xf64>, vector<f64>
  return %out#0, %out#1 : vector<16xf64>, vector<f64>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.scan
// CHECK-NOT: iree_gpu.subgroup_scan
