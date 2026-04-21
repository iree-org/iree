// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

// Test 1: Single (b,o) inclusive scan.
// Layout: subgroup=1, batch=1, outer=1, thread=4, element=4 along scan dim.
// Full vector is 16 elements along scan dim (1*1*1*4*4).

#layout_scan_1d = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [4],

  subgroup_strides = [1],
  thread_strides   = [1]
>

// CHECK-LABEL: @scan_single_bo_inclusive
func.func @scan_single_bo_inclusive(%src: vector<16xf32>, %init: vector<f32>) -> (vector<16xf32>, vector<f32>) {
  %src_l = iree_vector_ext.to_layout %src to layout(#layout_scan_1d) : vector<16xf32>
  %out:2 = vector.scan <add>, %src_l, %init {inclusive = true, reduction_dim = 0 : i64}
    : vector<16xf32>, vector<f32>
  return %out#0, %out#1 : vector<16xf32>, vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: %[[ID_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
// CHECK-DAG: %[[SRC_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16xf32> -> vector<1x1x4xf32>
// Local inclusive scan with identity init.
// CHECK: %[[LOCAL_SCAN:.*]], %[[LOCAL_TOTAL:.*]] = vector.scan <add>, %[[SRC_DIST]], %[[ID_VEC]] {inclusive = true, reduction_dim = 2 : i64} : vector<1x1x4xf32>, vector<1x1xf32>
// Subgroup scan of localTotal.
// CHECK: %[[SCALAR_TOTAL:.*]] = vector.extract %[[LOCAL_TOTAL]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SUBGROUP_SCAN:.*]], %[[SUBGROUP_TOTAL:.*]] = iree_gpu.subgroup_scan(%[[SCALAR_TOTAL]], {{.*}}) cluster(size = 4)
// Broadcast both subgroup scan results.
// CHECK-DAG: %[[SCAN_VEC:.*]] = vector.broadcast %[[SUBGROUP_SCAN]] : f32 to vector<1x1xf32>
// CHECK-DAG: %[[TOTAL_VEC:.*]] = vector.broadcast %[[SUBGROUP_TOTAL]] : f32 to vector<1x1xf32>
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[BLOCK_INCR:.*]] = arith.addf %[[SCAN_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[RESULT:.*]] = arith.addf %[[BLOCK_INCR_BCAST]], %[[LOCAL_SCAN]] : vector<1x1x4xf32>
// Accumulated value = batchOuterRunning (uniform), shape_cast to init shape.
// CHECK: %[[BO_RUNNING:.*]] = arith.addf %[[TOTAL_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// CHECK: %[[ACC:.*]] = vector.shape_cast %[[BO_RUNNING]] : vector<1x1xf32> to vector<f32>
// CHECK: iree_vector_ext.to_simd %[[ACC]] : vector<f32> -> vector<f32>
// CHECK: iree_vector_ext.to_simd %[[RESULT]] : vector<1x1x4xf32> -> vector<16xf32>

// -----

// Test 2: Single (b,o) exclusive scan.

#layout_scan_1d_excl = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [1],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [4],

  subgroup_strides = [1],
  thread_strides   = [1]
>

// CHECK-LABEL: @scan_single_bo_exclusive
func.func @scan_single_bo_exclusive(%src: vector<16xf32>, %init: vector<f32>) -> (vector<16xf32>, vector<f32>) {
  %src_l = iree_vector_ext.to_layout %src to layout(#layout_scan_1d_excl) : vector<16xf32>
  %out:2 = vector.scan <add>, %src_l, %init {inclusive = false, reduction_dim = 0 : i64}
    : vector<16xf32>, vector<f32>
  return %out#0, %out#1 : vector<16xf32>, vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: %[[ID_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
// CHECK-DAG: %[[SRC_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<16xf32> -> vector<1x1x4xf32>
// CHECK-DAG: %[[INIT_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<f32> -> vector<f32>
// Local exclusive scan with identity init.
// CHECK: %[[LOCAL_SCAN:.*]], %[[ACC_VAL:.*]] = vector.scan <add>, %[[SRC_DIST]], %[[ID_VEC]] {inclusive = false, reduction_dim = 2 : i64}
// Fix up localTotal: combine accumulated_value with last source element.
// CHECK: %[[LAST_ELEM:.*]] = vector.extract_strided_slice %[[SRC_DIST]] {offsets = [0, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]}
// CHECK: %[[LAST_FLAT:.*]] = vector.shape_cast %[[LAST_ELEM]] : vector<1x1x1xf32> to vector<1x1xf32>
// CHECK: %[[LOCAL_TOTAL:.*]] = arith.addf %[[ACC_VAL]], %[[LAST_FLAT]] : vector<1x1xf32>
// Subgroup scan of fixed-up localTotal.
// CHECK: %[[SCALAR_TOTAL:.*]] = vector.extract %[[LOCAL_TOTAL]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SUBGROUP_SCAN:.*]], %{{.*}} = iree_gpu.subgroup_scan(%[[SCALAR_TOTAL]], {{.*}}) cluster(size = 4)
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[SCAN_VEC:.*]] = vector.broadcast %[[SUBGROUP_SCAN]] : f32 to vector<1x1xf32>
// CHECK: %[[BLOCK_INCR:.*]] = arith.addf %[[SCAN_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[LOCAL_RESULT:.*]] = arith.addf %[[BLOCK_INCR_BCAST]], %[[LOCAL_SCAN]] : vector<1x1x4xf32>
// Application of user init via broadcast + combine.
// CHECK: %[[INIT_BCAST:.*]] = vector.broadcast %[[INIT_DIST]] : vector<f32> to vector<1x1x4xf32>
// CHECK: %[[RESULT:.*]] = arith.addf %[[INIT_BCAST]], %[[LOCAL_RESULT]] : vector<1x1x4xf32>
// Accumulated value: extract last element from result, broadcast from last thread.
// CHECK: %[[LAST_ACC:.*]] = vector.extract %[[RESULT]][0, 0, 3] : f32 from vector<1x1x4xf32>
// CHECK: %[[SHUFFLED:.*]], %{{.*}} = gpu.shuffle idx %[[LAST_ACC]], %{{.*}}, %{{.*}} : f32
// CHECK: %[[ACC_BCAST:.*]] = vector.broadcast %[[SHUFFLED]] : f32 to vector<f32>
// CHECK: iree_vector_ext.to_simd %[[ACC_BCAST]] : vector<f32> -> vector<f32>
// CHECK: iree_vector_ext.to_simd %[[RESULT]] : vector<1x1x4xf32> -> vector<16xf32>

// -----

// Test 3: Multi (b,o) inclusive scan.
// Layout: batch=2 along scan dim -> two (b,o) iterations.

#layout_scan_multi = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [2],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [4],

  subgroup_strides = [1],
  thread_strides   = [1]
>

// CHECK-LABEL: @scan_multi_bo_inclusive
func.func @scan_multi_bo_inclusive(%src: vector<32xf32>, %init: vector<f32>) -> (vector<32xf32>, vector<f32>) {
  %src_l = iree_vector_ext.to_layout %src to layout(#layout_scan_multi) : vector<32xf32>
  %out:2 = vector.scan <add>, %src_l, %init {inclusive = true, reduction_dim = 0 : i64}
    : vector<32xf32>, vector<f32>
  return %out#0, %out#1 : vector<32xf32>, vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: %[[ID_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
// CHECK-DAG: %[[SRC_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32xf32> -> vector<2x1x4xf32>
// First (b=0): extract srcChunk, local scan, subgroup scan.
// CHECK: %[[CHUNK0:.*]] = vector.extract %[[SRC_DIST]][0, 0] : vector<4xf32> from vector<2x1x4xf32>
// CHECK: %[[CHUNK0_RS:.*]] = vector.shape_cast %[[CHUNK0]] : vector<4xf32> to vector<1x1x4xf32>
// CHECK: %[[SCAN0:.*]], %[[TOTAL0:.*]] = vector.scan <add>, %[[CHUNK0_RS]], %[[ID_VEC]] {inclusive = true, reduction_dim = 2 : i64} : vector<1x1x4xf32>, vector<1x1xf32>
// CHECK: %[[SCALAR0:.*]] = vector.extract %[[TOTAL0]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SG_SCAN0:.*]], %[[SG_TOTAL0:.*]] = iree_gpu.subgroup_scan(%[[SCALAR0]], {{.*}}) cluster(size = 4)
// Broadcast both subgroup scan results.
// CHECK-DAG: %[[SG_SCAN0_VEC:.*]] = vector.broadcast %[[SG_SCAN0]] : f32 to vector<1x1xf32>
// CHECK-DAG: %[[SG_TOTAL0_VEC:.*]] = vector.broadcast %[[SG_TOTAL0]] : f32 to vector<1x1xf32>
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[BLOCK_INCR0:.*]] = arith.addf %[[SG_SCAN0_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR0_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR0]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[LOCAL_RESULT0:.*]] = arith.addf %[[BLOCK_INCR0_BCAST]], %[[SCAN0]] : vector<1x1x4xf32>
// CHECK: %[[RES0:.*]] = vector.insert_strided_slice %[[LOCAL_RESULT0]], %{{.*}} {offsets = [0, 0, 0], strides = [1, 1, 1]}
// Advance batchOuterRunning.
// CHECK: %[[BO_RUNNING1:.*]] = arith.addf %[[SG_TOTAL0_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// Second (b=1): extract srcChunk, local scan, subgroup scan.
// CHECK: %[[CHUNK1:.*]] = vector.extract %[[SRC_DIST]][1, 0] : vector<4xf32> from vector<2x1x4xf32>
// CHECK: %[[CHUNK1_RS:.*]] = vector.shape_cast %[[CHUNK1]] : vector<4xf32> to vector<1x1x4xf32>
// CHECK: %[[SCAN1:.*]], %[[TOTAL1:.*]] = vector.scan <add>, %[[CHUNK1_RS]], %[[ID_VEC]] {inclusive = true, reduction_dim = 2 : i64} : vector<1x1x4xf32>, vector<1x1xf32>
// CHECK: %[[SCALAR1:.*]] = vector.extract %[[TOTAL1]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SG_SCAN1:.*]], %[[SG_TOTAL1:.*]] = iree_gpu.subgroup_scan(%[[SCALAR1]], {{.*}}) cluster(size = 4)
// Broadcast both subgroup scan results.
// CHECK-DAG: %[[SG_SCAN1_VEC:.*]] = vector.broadcast %[[SG_SCAN1]] : f32 to vector<1x1xf32>
// CHECK-DAG: %[[SG_TOTAL1_VEC:.*]] = vector.broadcast %[[SG_TOTAL1]] : f32 to vector<1x1xf32>
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[BLOCK_INCR1:.*]] = arith.addf %[[BO_RUNNING1]], %[[SG_SCAN1_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR1_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR1]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[LOCAL_RESULT1:.*]] = arith.addf %[[BLOCK_INCR1_BCAST]], %[[SCAN1]] : vector<1x1x4xf32>
// CHECK: %[[RESULT:.*]] = vector.insert_strided_slice %[[LOCAL_RESULT1]], %[[RES0]] {offsets = [1, 0, 0], strides = [1, 1, 1]}
// Accumulated value = final batchOuterRunning.
// CHECK: %[[FINAL_BO_RUNNING:.*]] = arith.addf %[[BO_RUNNING1]], %[[SG_TOTAL1_VEC]] : vector<1x1xf32>
// CHECK: %[[ACC:.*]] = vector.shape_cast %[[FINAL_BO_RUNNING]] : vector<1x1xf32> to vector<f32>
// CHECK: iree_vector_ext.to_simd %[[ACC]] : vector<f32> -> vector<f32>
// CHECK: iree_vector_ext.to_simd %[[RESULT]] : vector<2x1x4xf32> -> vector<32xf32>

// -----

// Test 4: Multi (b,o) exclusive scan.

#layout_scan_multi_excl = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile    = [2],
  outer_tile    = [1],
  thread_tile   = [4],
  element_tile  = [4],

  subgroup_strides = [1],
  thread_strides   = [1]
>

// CHECK-LABEL: @scan_multi_bo_exclusive
func.func @scan_multi_bo_exclusive(%src: vector<32xf32>, %init: vector<f32>) -> (vector<32xf32>, vector<f32>) {
  %src_l = iree_vector_ext.to_layout %src to layout(#layout_scan_multi_excl) : vector<32xf32>
  %out:2 = vector.scan <add>, %src_l, %init {inclusive = false, reduction_dim = 0 : i64}
    : vector<32xf32>, vector<f32>
  return %out#0, %out#1 : vector<32xf32>, vector<f32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: %[[ID_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<1x1xf32>
// CHECK-DAG: %[[SRC_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32xf32> -> vector<2x1x4xf32>
// CHECK-DAG: %[[INIT_DIST:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<f32> -> vector<f32>
// First (b=0): extract srcChunk, local exclusive scan.
// CHECK: %[[CHUNK0:.*]] = vector.extract %[[SRC_DIST]][0, 0] : vector<4xf32> from vector<2x1x4xf32>
// CHECK: %[[CHUNK0_RS:.*]] = vector.shape_cast %[[CHUNK0]] : vector<4xf32> to vector<1x1x4xf32>
// CHECK: %[[SCAN0:.*]], %[[ACC0:.*]] = vector.scan <add>, %[[CHUNK0_RS]], %[[ID_VEC]] {inclusive = false, reduction_dim = 2 : i64}
// Fix up localTotal: combine accumulated_value with last source element.
// CHECK: %[[LAST0:.*]] = vector.extract_strided_slice %[[CHUNK0_RS]] {offsets = [0, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]}
// CHECK: %[[LAST0_FLAT:.*]] = vector.shape_cast %[[LAST0]] : vector<1x1x1xf32> to vector<1x1xf32>
// CHECK: %[[LOCAL_TOTAL0:.*]] = arith.addf %[[ACC0]], %[[LAST0_FLAT]] : vector<1x1xf32>
// Subgroup scan.
// CHECK: %[[SCALAR0:.*]] = vector.extract %[[LOCAL_TOTAL0]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SG_SCAN0:.*]], %[[SG_TOTAL0:.*]] = iree_gpu.subgroup_scan(%[[SCALAR0]], {{.*}}) cluster(size = 4)
// Broadcast both subgroup scan results.
// CHECK-DAG: %[[SG_SCAN0_VEC:.*]] = vector.broadcast %[[SG_SCAN0]] : f32 to vector<1x1xf32>
// CHECK-DAG: %[[SG_TOTAL0_VEC:.*]] = vector.broadcast %[[SG_TOTAL0]] : f32 to vector<1x1xf32>
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[BLOCK_INCR0:.*]] = arith.addf %[[SG_SCAN0_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR0_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR0]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[LOCAL_RESULT0:.*]] = arith.addf %[[BLOCK_INCR0_BCAST]], %[[SCAN0]] : vector<1x1x4xf32>
// CHECK: %[[RES0:.*]] = vector.insert_strided_slice %[[LOCAL_RESULT0]], %{{.*}} {offsets = [0, 0, 0], strides = [1, 1, 1]}
// Advance batchOuterRunning.
// CHECK: %[[BO_RUNNING1:.*]] = arith.addf %[[SG_TOTAL0_VEC]], %[[ID_VEC]] : vector<1x1xf32>
// Second (b=1): extract srcChunk, local exclusive scan.
// CHECK: %[[CHUNK1:.*]] = vector.extract %[[SRC_DIST]][1, 0] : vector<4xf32> from vector<2x1x4xf32>
// CHECK: %[[CHUNK1_RS:.*]] = vector.shape_cast %[[CHUNK1]] : vector<4xf32> to vector<1x1x4xf32>
// CHECK: %[[SCAN1:.*]], %[[ACC1:.*]] = vector.scan <add>, %[[CHUNK1_RS]], %[[ID_VEC]] {inclusive = false, reduction_dim = 2 : i64}
// Fix up localTotal for second chunk.
// CHECK: %[[LAST1:.*]] = vector.extract_strided_slice %[[CHUNK1_RS]] {offsets = [0, 0, 3], sizes = [1, 1, 1], strides = [1, 1, 1]}
// CHECK: %[[LAST1_FLAT:.*]] = vector.shape_cast %[[LAST1]] : vector<1x1x1xf32> to vector<1x1xf32>
// CHECK: %[[LOCAL_TOTAL1:.*]] = arith.addf %[[ACC1]], %[[LAST1_FLAT]] : vector<1x1xf32>
// Subgroup scan.
// CHECK: %[[SCALAR1:.*]] = vector.extract %[[LOCAL_TOTAL1]][0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[SG_SCAN1:.*]], %{{.*}} = iree_gpu.subgroup_scan(%[[SCALAR1]], {{.*}}) cluster(size = 4)
// blockIncrement = combine(subgroupScan, batchOuterRunning), apply to localScan.
// CHECK: %[[SG_SCAN1_VEC:.*]] = vector.broadcast %[[SG_SCAN1]] : f32 to vector<1x1xf32>
// CHECK: %[[BLOCK_INCR1:.*]] = arith.addf %[[BO_RUNNING1]], %[[SG_SCAN1_VEC]] : vector<1x1xf32>
// CHECK: %[[BLOCK_INCR1_BCAST:.*]] = vector.broadcast %[[BLOCK_INCR1]] : vector<1x1xf32> to vector<1x1x4xf32>
// CHECK: %[[LOCAL_RESULT1:.*]] = arith.addf %[[BLOCK_INCR1_BCAST]], %[[SCAN1]] : vector<1x1x4xf32>
// CHECK: %[[PRE_INIT:.*]] = vector.insert_strided_slice %[[LOCAL_RESULT1]], %[[RES0]] {offsets = [1, 0, 0], strides = [1, 1, 1]}
// Application of user init.
// CHECK: %[[INIT_BCAST:.*]] = vector.broadcast %[[INIT_DIST]] : vector<f32> to vector<2x1x4xf32>
// CHECK: %[[RESULT:.*]] = arith.addf %[[INIT_BCAST]], %[[PRE_INIT]] : vector<2x1x4xf32>
// Accumulated value: extract last element from result, broadcast from last thread.
// CHECK: %[[LAST_ACC:.*]] = vector.extract %[[RESULT]][1, 0, 3] : f32 from vector<2x1x4xf32>
// CHECK: %[[SHUFFLED:.*]], %{{.*}} = gpu.shuffle idx %[[LAST_ACC]], %{{.*}}, %{{.*}} : f32
// CHECK: %[[ACC_BCAST:.*]] = vector.broadcast %[[SHUFFLED]] : f32 to vector<f32>
// CHECK: iree_vector_ext.to_simd %[[ACC_BCAST]] : vector<f32> -> vector<f32>
// CHECK: iree_vector_ext.to_simd %[[RESULT]] : vector<2x1x4xf32> -> vector<32xf32>
