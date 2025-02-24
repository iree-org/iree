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
// CHECK: %[[MASK:.+]] = vector.create_mask %[[SELTREE3]], %c8 : vector<8x8xi1>

// CHECK: %[[MASK_EXTR:.+]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<8x8xi1> to vector<2x8xi1>
// CHECK: %[[READ:.+]] = vector.transfer_read %arg0{{.*}}, %[[MASK_EXTR]] {in_bounds = [true, true]} : memref<?x128xf16>, vector<2x8xf16>
// CHECK: vector.transfer_write %[[READ]], %arg1{{.*}}, %[[MASK_EXTR]] {in_bounds = [true, true]} : vector<2x8xf16>, memref<?x128xf16>

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

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask %c8, {{.*}}, %c1 : vector<8x8x1xi1>
// CHECK: %[[DISTR_MASK_0:.+]] = vector.extract_strided_slice %[[DISTR_MASK]] {offsets = [0, 0, 0], sizes = [8, 2, 1], strides = [1, 1, 1]} : vector<8x8x1xi1> to vector<8x2x1xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK_0]]

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

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask %c8, {{.*}} : vector<8x8xi1>
// CHECK: %[[DISTR_MASK_0:.+]] = vector.extract_strided_slice %[[DISTR_MASK]] {offsets = [0, 0], sizes = [8, 2], strides = [1, 1]} : vector<8x8xi1> to vector<8x2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK_0]]

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

// CHECK: %[[DISTR_MASK:.+]] = vector.create_mask {{.*}} : vector<8xi1>
// CHECK: %[[DISTR_MASK_0:.+]] = vector.extract_strided_slice %16 {offsets = [0], sizes = [2], strides = [1]} : vector<8xi1> to vector<2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[DISTR_MASK_0]]

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
// CHECK: %[[MASK_PCK:.+]] = vector.shape_cast %[[MASK]] : vector<8x8xi1> to vector<2x2x2x1x1x8xi1>
// CHECK: %[[MASK_ITL_PCK:.+]] = vector.transpose %[[MASK_PCK]], [0, 3, 1, 4, 2, 5] : vector<2x2x2x1x1x8xi1> to vector<2x1x2x1x2x8xi1>

// CHECK: %[[SELECT:.+]] = arith.select %[[MASK_ITL_PCK]], {{.*}}, %[[RED_IDENTITY]] : vector<2x1x2x1x2x8xi1>, vector<2x1x2x1x2x8xf16>
// CHECK: vector.multi_reduction <add>, %[[SELECT]], {{.*}} [0, 2, 4] : vector<2x1x2x1x2x8xf16> to vector<1x1x8xf16>
