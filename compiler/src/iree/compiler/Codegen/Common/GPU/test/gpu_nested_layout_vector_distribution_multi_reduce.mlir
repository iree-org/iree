// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize -mlir-print-local-scope --cse %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  // We are reducing along dim=1, so each thread will reduce
  // 2 batches x 4 elements = 8 elements.
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  // We are reducing on dim=1, which is distributed over 4 threads. Based
  // on the subgroup basis and thread order, the shuffle offset is 16.
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @mfma_16x16x16_out_reduced_dim1(%arg0: vector<32x32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = vector.multi_reduction <maximumf>, %arg0l, %arg1 [1] : vector<32x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @mfma_16x16x16_out_reduced_dim1
// CHECK-DAG: %[[IDENTITY:.*]] = arith.constant dense<0xFF800000> : vector<2x1x1xf32>
// CHECK-DAG: %[[DARG0:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32x32xf32> -> vector<2x2x1x1x1x4xf32>
// CHECK-DAG: %[[DARG1:.*]] = iree_vector_ext.to_simt %{{.*}} : vector<32xf32> -> vector<2x1x1xf32>
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %[[DARG0]], %[[IDENTITY]] [1, 3, 5] : vector<2x2x1x1x1x4xf32> to vector<2x1x1xf32>
// Global reduction
// CHECK: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// Accumulator reduction
// CHECK: %[[ACC_REDUC:.+]] = arith.maximumf %{{.*}}, %[[DARG1]] : vector<2x1x1xf32>
// CHECK: iree_vector_ext.to_simd %[[ACC_REDUC]] : vector<2x1x1xf32> -> vector<32xf32>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  // We are reducing along dim=1, so each thread will reduce
  // 4 batches x 4 elements = 16 elements.
  batch_tile    = [1, 4],
  outer_tile        = [1, 1],
  // We are reducing on dim=1, which is distributed over 2 threads. Based
  // on the subgroup basis and thread order, the shuffle offset is 32.
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

func.func @mfma_32x32x8_out_reduced_dim1(%arg0: vector<32x32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = vector.multi_reduction <maximumf>, %arg0l, %arg1 [1] : vector<32x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @mfma_32x32x8_out_reduced_dim1
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<1x4x1x1x1x4xf32> to vector<1x1x1xf32>
// Global reduction
// CHECK: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 2, stride = 32) : (f32) -> f32
// Accumulator reduction
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<1x1x1xf32>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [1, 1],
  thread_strides = [1, 16]
>

func.func @mfma_16x16x16_out_reduced_alldims(%arg0: vector<32x32xf16>, %arg1: f16) -> f16 {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf16>
  %0 = vector.multi_reduction <maximumf>, %arg0l, %arg1 [0, 1] : vector<32x32xf16> to f16
  return %0 : f16
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @mfma_16x16x16_out_reduced_alldims
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5] : vector<2x2x1x1x1x4xf16> to f16
// Global reduction
// CHECK: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 16) : (f16) -> f16
// CHECK-NEXT: gpu.subgroup_reduce maximumf %{{.*}} cluster(size = 4, stride = 16) : (f16) -> f16
// Accumulator reduction
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<1xf16>

// -----

#nested = #iree_vector_ext.nested_layout<
  // There will two partial reductions across
  // two subgroups.
  subgroup_tile = [1, 2],
  // We are reducing along dim=1, so each thread will reduce
  // 1 batches x 4 elements = 4 elements.
  batch_tile = [2, 1],
  outer_tile = [1, 1],
  // We are reducing on dim=1, which is distributed over 4 threads. Based
  // on the subgroup basis and thread order, the shuffle offset is 16.
  thread_tile = [16, 4],
  element_tile = [1, 4],

  subgroup_strides = [2, 1],
  thread_strides = [1, 16]
>

func.func @inter_subgroup_reduction(%arg0: vector<32x32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> {
  %arg0l = iree_vector_ext.to_layout %arg0 to layout(#nested) : vector<32x32xf32>
  %0 = vector.multi_reduction <maximumf>, %arg0l, %arg1 [1] : vector<32x32xf32> to vector<32xf32>
  return %0 : vector<32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @inter_subgroup_reduction
// CHECK-DAG: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<2x1x1x1x1x2xf32>
// CHECK-DAG: %[[CST1:.+]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// Local reduction
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [1, 3, 5] : vector<2x1x1x1x1x4xf32> to vector<2x1x1xf32>
// Thread reduction
// CHECK: %[[THREAD_RED0:.+]] = gpu.subgroup_reduce  maximumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// CHECK: %[[THREAD_RED1:.+]] = vector.insert %[[THREAD_RED0]], %cst_1 [0] : f32 into vector<2xf32>
// CHECK: %[[THREAD_RED2:.+]] = gpu.subgroup_reduce  maximumf %{{.*}} cluster(size = 4, stride = 16) : (f32) -> f32
// CHECK: %[[THREAD_RED3:.+]] = vector.insert %[[THREAD_RED2]], %[[THREAD_RED1]] [1] : f32 into vector<2xf32>
// CHECK: %[[THREAD_RED4:.+]] = vector.shape_cast %[[THREAD_RED3]] : vector<2xf32> to vector<2x1x1xf32>
// Subgroup reduction
// CHECK-DAG: %[[ALLOC:.+]] = memref.alloc() : memref<32x2xf32, #gpu.address_space<workgroup>>
// CHECK: gpu.barrier
// CHECK-DAG: %[[TIDX0:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%thread_id_x]
// CHECK-DAG: %[[TIDX1:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 16 + 16)>()[%thread_id_x]
// CHECK-DAG: %[[SGIDX:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) mod 2)>()[%thread_id_x]
// CHECK-DAG: %[[EXTRACT0:.+]] = vector.extract %[[THREAD_RED4]][0] : vector<1x1xf32> from vector<2x1x1xf32>
// CHECK-DAG: %[[EXTRACT1:.+]] = vector.extract %[[THREAD_RED4]][1] : vector<1x1xf32> from vector<2x1x1xf32>
// CHECK-DAG: vector.transfer_write %[[EXTRACT0]], %[[ALLOC]][%[[TIDX0]], %[[SGIDX]]]
// CHECK-DAG: vector.transfer_write %[[EXTRACT1]], %[[ALLOC]][%[[TIDX1]], %[[SGIDX]]]
// CHECK: gpu.barrier
// CHECK-DAG: %[[READ0:.+]] = vector.transfer_read %alloc[%[[TIDX0]], %c0], {{.*}} {in_bounds = [false, true]} : memref<32x2xf32, #gpu.address_space<workgroup>>, vector<1x2xf32>
// CHECK-DAG: %[[GATHER0:.+]] = vector.insert_strided_slice %[[READ0]], %[[CST]] {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<2x1x1x1x1x2xf32>
// CHECK-DAG: %[[READ1:.+]] = vector.transfer_read %alloc[%[[TIDX1]], %c0], %cst_0 {in_bounds = [false, true]} : memref<32x2xf32, #gpu.address_space<workgroup>>, vector<1x2xf32>
// CHECK-DAG: %[[GATHER1:.+]] = vector.insert_strided_slice %[[READ1]], %[[GATHER0]] {offsets = [1, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf32> into vector<2x1x1x1x1x2xf32>
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_simt %arg1 : vector<32xf32> -> vector<2x1x1xf32>
// CHECK-DAG: %[[SGRED:.+]] = vector.multi_reduction <maximumf>, %[[GATHER1]], {{.*}} [1, 3, 5] : vector<2x1x1x1x1x2xf32> to vector<2x1x1xf32>
// CHECK-DAG: arith.maximumf %[[SGRED]], %[[ACC]] : vector<2x1x1xf32>
