// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse --canonicalize --mlir-print-local-scope %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [2, 1],
  outer_tile = [2, 1],
  thread_tile = [16, 16],
  element_tile = [2, 8],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @masked_read_write(%arg0 : memref<?x128xf16>, %arg1 : memref<?x128xf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %dyn = memref.dim %arg0, %c0 : memref<?x128xf16>
  %41 = vector.create_mask %dyn, %c128 : vector<256x128xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0], %cst_6, %41 {in_bounds = [true, true]} : memref<?x128xf16>, vector<256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<256x128xf16>
  vector.transfer_write %43, %arg1[%c0, %c0], %41 {in_bounds = [true, true]} : vector<256x128xf16>, memref<?x128xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write
// CHECK: %[[DIM:.+]] = memref.dim %arg0, %c0 : memref<?x128xf16>
// CHECK: %[[VSID:.+]]:3 = affine.delinearize_index %thread_id_x into (2, 64) : index, index, index
// CHECK: %[[VTID:.+]]:3 = affine.delinearize_index %thread_id_x into (16, 16) : index, index, index
// CHECK: %[[LASTIDX:.+]] = arith.subi %[[DIM]], %c1 : index
// CHECK: %[[PACKED_LASTIDX:.+]]:4 = affine.delinearize_index %[[LASTIDX]] into (2, 4, 16, 2) : index, index, index, index

// CHECK: %[[ETILE_VALID:.+]] = affine.linearize_index [%[[PACKED_LASTIDX]]#1, %c1] by (4, 2) : index
// CHECK: %[[ETILE_VALID_BOUND:.+]] = arith.addi %[[ETILE_VALID]], %c1 : index
// CHECK: %[[DISTR_LASTIDX:.+]] = affine.linearize_index [%[[PACKED_LASTIDX]]#1, %[[PACKED_LASTIDX]]#3] by (4, 2) : index
// CHECK: %[[DISTR_BOUND:.+]] = arith.addi %[[DISTR_LASTIDX]], %c1 : index

// CHECK: %[[EQ_BOUND_TID:.+]] = arith.cmpi eq, %[[VTID]]#1, %[[PACKED_LASTIDX]]#2 : index
// CHECK: %[[LT_BOUND_TID:.+]] = arith.cmpi slt, %[[VTID]]#1, %[[PACKED_LASTIDX]]#2 : index
// CHECK: %[[EQ_BOUND_SID:.+]] = arith.cmpi eq, %[[VSID]]#1, %[[PACKED_LASTIDX]]#0 : index
// CHECK: %[[LT_BOUND_SID:.+]] = arith.cmpi slt, %[[VSID]]#1, %[[PACKED_LASTIDX]]#0 : index

// CHECK: %[[SELTREE0:.+]] = arith.select %[[LT_BOUND_TID]], %[[ETILE_VALID_BOUND]], %c0 : index
// CHECK: %[[SELTREE1:.+]] = arith.select %[[EQ_BOUND_TID]], %[[DISTR_BOUND]], %[[SELTREE0]] : index
// CHECK: %[[SELTREE2:.+]] = arith.select %[[LT_BOUND_SID]], %c8, %c0 : index
// CHECK: %[[SELTREE3:.+]] = arith.select %[[EQ_BOUND_SID]], %[[SELTREE1]], %[[SELTREE2]] : index
// CHECK: %[[MASK:.+]] = vector.create_mask %[[SELTREE3]], %c8 : vector<2x8xi1>

// CHECK: %[[READ:.+]] = vector.transfer_read %arg0{{.*}}, %[[MASK]] {in_bounds = [true, true]} : memref<?x128xf16>, vector<2x8xf16>
// CHECK: vector.transfer_write %[[READ]], %arg1{{.*}}, %[[MASK]] {in_bounds = [true, true]} : vector<2x8xf16>, memref<?x128xf16>

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 2, 1],
  batch_tile = [1, 2, 1],
  outer_tile = [1, 2, 1],
  thread_tile = [1, 16, 16],
  element_tile = [1, 2, 8],

  subgroup_strides = [1, 1, 0],
  thread_strides = [1, 16, 1]
>

func.func @masked_read_write_perm(%arg0 : memref<128x?x1xf16>, %arg1 : memref<128x?x1xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %dyn = memref.dim %arg0, %c1 : memref<128x?x1xf16>
  %41 = vector.create_mask %c128, %dyn, %c1 : vector<128x256x1xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst_6, %41 {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1, d0)>} : memref<128x?x1xf16>, vector<1x256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<1x256x128xf16>
  vector.transfer_write %43, %arg1[%c0, %c0, %c0], %41 {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1, d0)>} : vector<1x256x128xf16>, memref<128x?x1xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_perm

// Here we check the layout enforcement was carried out
// accounting for permutation

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask %c8, {{.*}}, %c1 : vector<8x2x1xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK]]

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [2, 1],
  outer_tile = [2, 1],
  thread_tile = [16, 16],
  element_tile = [2, 8],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @masked_read_write_perm_minor(%arg0 : memref<128x?x1xf16>, %arg1 : memref<128x?x1xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %dyn = memref.dim %arg0, %c1 : memref<128x?x1xf16>
  %41 = vector.create_mask %c128, %dyn : vector<128x256xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst_6, %41 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : memref<128x?x1xf16>, vector<256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<256x128xf16>
  vector.transfer_write %43, %arg1[%c0, %c0, %c0], %41 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} : vector<256x128xf16>, memref<128x?x1xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_perm_minor

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask %c8, {{.*}} : vector<8x2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK]]

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [2, 1],
  outer_tile = [2, 1],
  thread_tile = [16, 16],
  element_tile = [2, 8],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @masked_read_write_perm_bcast(%arg0 : memref<128x?x1xf16>, %arg1 : memref<128x?x1xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %dyn = memref.dim %arg0, %c1 : memref<128x?x1xf16>
  %41 = vector.create_mask %dyn : vector<256xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst_6, %41 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, 0)>} : memref<128x?x1xf16>, vector<256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<256x128xf16>
  %44 = vector.create_mask %dyn, %c128 : vector<256x128xi1>
  vector.transfer_write %43, %arg1[%c0, %c0, %c0], %44 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>} : vector<256x128xf16>, memref<128x?x1xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_perm_bcast

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask {{.*}} : vector<2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK]]

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [2, 1],
  outer_tile = [2, 1],
  thread_tile = [16, 16],
  element_tile = [2, 8],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @masked_read_write_reduce(%arg0 : memref<?x128xf16>, %arg1 : memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant dense<0.000000e+00> : vector<128xf16>

  %dyn = memref.dim %arg0, %c0 : memref<?x128xf16>
  %41 = vector.create_mask %dyn, %c128 : vector<256x128xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0], %cst_6, %41 {in_bounds = [true, true]} : memref<?x128xf16>, vector<256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<256x128xf16>
  %44 = vector.mask %41 { vector.multi_reduction <add>, %43, %cst_1 [0] : vector<256x128xf16> to vector<128xf16> } : vector<256x128xi1> -> vector<128xf16>
  vector.transfer_write %44, %arg1[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_reduce

// CHECK: %[[RED_IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<2x1x2x1x2x8xf16>

// CHECK: %[[MASK:.+]] = vector.create_mask
// CHECK: %[[MASK_PCK:.+]] = vector.shape_cast %[[MASK]] : vector<8x8xi1> to vector<2x1x2x1x2x8xi1>

// CHECK: %[[SELECT:.+]] = arith.select %[[MASK_PCK]], {{.*}}, %[[RED_IDENTITY]] : vector<2x1x2x1x2x8xi1>, vector<2x1x2x1x2x8xf16>
// CHECK: vector.multi_reduction <add>, %[[SELECT]], {{.*}} [0, 2, 4] : vector<2x1x2x1x2x8xf16> to vector<1x1x8xf16>

// -----

#lhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [2],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [16],
  element_tile = [2],

  subgroup_strides = [2],
  thread_strides = [8]
>

#rhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 2],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [8, 16],
  element_tile = [2, 2],

  subgroup_strides = [1, 2],
  thread_strides = [1, 8]
>

#out = #iree_vector_ext.nested_layout<
  subgroup_tile = [2],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [8],
  element_tile = [2],

  subgroup_strides = [1],
  thread_strides = [1]
>

func.func @masked_read_write_contract(%arg0 : memref<?xf16>, %arg1 : memref<?x?xf16>, %arg2 : memref<?xf16>) {
  %c0 = arith.constant 0 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %acc = arith.constant dense<0.000000e+00> : vector<32xf16>

  %reddim = memref.dim %arg0, %c0 : memref<?xf16>
  %pardim = memref.dim %arg1, %c0 : memref<?x?xf16>
  %arg0mask = vector.create_mask %reddim :  vector<64xi1>
  %arg1mask = vector.create_mask %pardim, %reddim :  vector<32x64xi1>
  %arg2mask = vector.create_mask %pardim :  vector<32xi1>
  %opmask = vector.create_mask %reddim, %pardim :  vector<64x32xi1>

  %arg0read = vector.transfer_read %arg0[%c0], %cst_6, %arg0mask {in_bounds = [true]} : memref<?xf16>, vector<64xf16>
  %arg0readl = iree_vector_ext.to_layout %arg0read to layout(#lhs) : vector<64xf16>
  %arg1read = vector.transfer_read %arg1[%c0, %c0], %cst_6, %arg1mask {in_bounds = [true, true]} : memref<?x?xf16>, vector<32x64xf16>
  %arg1readl = iree_vector_ext.to_layout %arg1read to layout(#rhs) : vector<32x64xf16>
  %gemm = vector.mask %opmask { vector.contract {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"], kind = #vector.kind<add>} %arg0readl, %arg1readl, %acc : vector<64xf16>, vector<32x64xf16> into vector<32xf16> } : vector<64x32xi1> -> vector<32xf16>
  %gemml = iree_vector_ext.to_layout %gemm to layout(#out) : vector<32xf16>
  vector.transfer_write %gemml, %arg2[%c0], %arg2mask {in_bounds = [true]} : vector<32xf16>, memref<?xf16>

  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_contract

// CHECK-DAG: %[[RED_IDENTITY_LHS:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2xf16>
// CHECK-DAG: %[[RED_IDENTITY_RHS:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x1x1x2x2xf16>

// Note this this transposed to match the second indexing map
// CHECK-DAG: %[[MASK_LHS:.+]] = vector.create_mask %[[LHSUB:.+]] : vector<2xi1>
// CHECK-DAG: %[[MASK_OP:.+]] = vector.create_mask %[[OPUB0:.+]], %[[LHSUB]] : vector<2x2xi1>

// Note MASK_OP_1D is equivalent to MASK_LHS.
// Currently, it does not fold away.
// CHECK-DAG: %[[MASK_OP_1D:.+]] = vector.extract %[[MASK_OP]][0] : vector<2xi1> from vector<2x2xi1>
// CHECK-DAG: %[[MASK_OP_1D_PACKED:.+]] = vector.shape_cast %[[MASK_OP_1D]] : vector<2xi1> to vector<1x1x2xi1>
// CHECK-DAG: %[[MASK_OP_PACKED:.+]] = vector.shape_cast %[[MASK_OP]] : vector<2x2xi1> to vector<1x1x1x1x2x2xi1>
// CHECK-DAG: %[[MASK_OUT:.+]] = vector.create_mask {{.*}} : vector<2xi1>

// CHECK-DAG: %[[LHS_READ:.+]] = vector.transfer_read %arg0{{.*}} %[[MASK_LHS]] {in_bounds = [true]} : memref<?xf16>, vector<2xf16>
// CHECK-DAG: %[[LHS:.+]] = vector.insert_strided_slice %[[LHS_READ]]
// CHECK-DAG: %[[RHS_READ:.+]] = vector.transfer_read %arg1{{.*}} %[[MASK_OP]] {in_bounds = [true, true]} : memref<?x?xf16>, vector<2x2xf16>
// CHECK-DAG: %[[RHS:.+]] = vector.insert_strided_slice %[[RHS_READ]]

// CHECK-DAG: %[[LHS_SELECT:.+]] = arith.select %[[MASK_OP_1D_PACKED]], %[[LHS]], %[[RED_IDENTITY_LHS]] : vector<1x1x2xi1>, vector<1x1x2xf16>
// CHECK-DAG: %[[RHS_SELECT:.+]] = arith.select %[[MASK_OP_PACKED]], %[[RHS]], %[[RED_IDENTITY_RHS]] : vector<1x1x1x1x2x2xi1>, vector<1x1x1x1x2x2xf16>

// CHECK: vector.contract {{.*}} %[[LHS_SELECT]], %[[RHS_SELECT]]

// -----

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [2, 1],
  outer_tile = [2, 1],
  thread_tile = [16, 16],
  element_tile = [2, 8],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @masked_read_write_unaligned(%arg0 : memref<17x128xf16>, %arg1 : memref<17x128xf16>) {
  %c0 = arith.constant 0 : index
  %cst_6 = arith.constant 0.000000e+00 : f16
  %41 = vector.constant_mask [17, 128] : vector<256x128xi1>
  %42 = vector.transfer_read %arg0[%c0, %c0], %cst_6, %41 {in_bounds = [true, true]} : memref<17x128xf16>, vector<256x128xf16>
  %43 = iree_vector_ext.to_layout %42 to layout(#nested) : vector<256x128xf16>
  vector.transfer_write %43, %arg1[%c0, %c0], %41 {in_bounds = [true, true]} : vector<256x128xf16>, memref<17x128xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_unaligned
// CHECK: %[[VSID:.+]]:3 = affine.delinearize_index %thread_id_x into (2, 64) : index, index, index
// CHECK: %[[VTID:.+]]:3 = affine.delinearize_index %thread_id_x into (16, 16) : index, index, index

// CHECK: %[[EQ_BOUND_TID:.+]] = arith.cmpi eq, %[[VTID]]#1, %c8 : index
// CHECK: %[[LT_BOUND_TID:.+]] = arith.cmpi slt, %[[VTID]]#1, %c8 : index
// CHECK: %[[EQ_BOUND_SID:.+]] = arith.cmpi eq, %[[VSID]]#1, %c0 : index
// CHECK: %[[LT_BOUND_SID:.+]] = arith.cmpi slt, %[[VSID]]#1, %c0 : index

// CHECK: %[[SELTREE0:.+]] = arith.select %[[LT_BOUND_TID]], %c2, %c0 : index
// CHECK: %[[SELTREE1:.+]] = arith.select %[[EQ_BOUND_TID]], %c1, %[[SELTREE0]] : index
// CHECK: %[[SELTREE2:.+]] = arith.select %[[LT_BOUND_SID]], %c8, %c0 : index
// CHECK: %[[SELTREE3:.+]] = arith.select %[[EQ_BOUND_SID]], %[[SELTREE1]], %[[SELTREE2]] : index
// CHECK: %[[MASK:.+]] = vector.create_mask %[[SELTREE3]], %c8 : vector<2x8xi1>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [4, 1],
  batch_tile = [4, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [1, 8],

  subgroup_strides = [1, 0],
  thread_strides = [0, 0]
>

func.func @paged_transfer_gather_mask(%indices: vector<16xindex>,
  %source: memref<4096x512x8xf16>) -> vector<16x8xf16> {

  %cst0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  %dim = memref.dim %source, %c0 : memref<4096x512x8xf16>
  %mask = vector.create_mask %c7, %c7 : vector<16x8xi1>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0, %c0]
  [None, %indices: vector<16xindex>, None], %cst0, %mask { indexed_maps = [
                                             affine_map<(d0, d1, d2) -> (d1)>],
    permutation_map = affine_map<(d0, d1, d2) -> (d1, d2)>,
    in_bounds = [true, true] }
  : memref<4096x512x8xf16>, vector<16x8xf16>

  %l_out = iree_vector_ext.to_layout %out to layout(#layout) : vector<16x8xf16>

  return %l_out : vector<16x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @paged_transfer_gather_mask
// CHECK: %[[MASK0:.+]] = vector.create_mask %{{.*}}, %c7 : vector<1x8xi1>
// CHECK: vector.transfer_read
// CHECK-SAME: %[[MASK0]]
// CHECK: %[[MASK1:.+]] = vector.create_mask %{{.*}}, %c7 : vector<1x8xi1>
// CHECK: vector.transfer_read
// CHECK-SAME: %[[MASK1]]
// CHECK: %[[MASK2:.+]] = vector.create_mask %{{.*}}, %c7 : vector<1x8xi1>
// CHECK: vector.transfer_read
// CHECK-SAME: %[[MASK2]]
// CHECK: %[[MASK3:.+]] = vector.create_mask %{{.*}}, %c7 : vector<1x8xi1>
// CHECK: vector.transfer_read
// CHECK-SAME: %[[MASK3]]
