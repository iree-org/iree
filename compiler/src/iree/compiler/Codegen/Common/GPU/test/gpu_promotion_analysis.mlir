// RUN: iree-opt --pass-pipeline="builtin.module(any(iree-codegen-test-gpu-promotion-analysis))" --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 16],
  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

func.func @direct_propagation(%src: memref<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %read = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
  %out = iree_vector_ext.to_layout %read to layout(#layout)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<16x16xf16>
  return %out : vector<16x16xf16>
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

func.func @through_elementwise(%src: memref<16x16xf16>, %other: vector<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %read = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %mul = arith.mulf %read, %other : vector<16x16xf16>
  %out = iree_vector_ext.to_layout %mul to layout(#layout)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<16x16xf16>
  return %out : vector<16x16xf16>
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

func.func @through_transpose(%src: memref<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %read = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %transpose = vector.transpose %read, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
  %out = iree_vector_ext.to_layout %transpose to layout(#layout)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<16x16xf16>
  return %out : vector<16x16xf16>
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

func.func @through_scf_for(%src: memref<16x16xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f16
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %init = vector.transfer_read %src[%c0, %c0], %cst : memref<16x16xf16>, vector<16x16xf16>
  // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
  %loop = scf.for %iv = %c0 to %c4 step %c1 iter_args(%arg = %init) -> vector<16x16xf16> {
    // expected-remark @below {{promotion type of result #0 is #iree_gpu.use_global_load_dma}}
    %update = arith.addf %arg, %arg : vector<16x16xf16>
    scf.yield %update : vector<16x16xf16>
  }
  %out = iree_vector_ext.to_layout %loop to layout(#layout)
      {iree_gpu.promotion_type = #iree_gpu.use_global_load_dma} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}
