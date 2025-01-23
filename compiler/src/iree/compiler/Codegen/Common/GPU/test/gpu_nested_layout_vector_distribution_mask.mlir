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
// CHECK: %[[VSID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 64) mod 2)>()[%thread_id_x]
// CHECK: %[[VTID:.+]] = affine.apply affine_map<()[s0] -> ((s0 floordiv 16) mod 16)>()[%thread_id_x]
// CHECK: %[[LASTIDX:.+]] = arith.subi %[[DIM]], %c1 : index
// CHECK: %[[PACKED_LASTIDX:.+]]:4 = affine.delinearize_index %[[LASTIDX]] into (2, 4, 16, 2) : index, index, index, index

// CHECK: %[[ETILE_VALID:.+]] = affine.linearize_index [%[[PACKED_LASTIDX]]#1, %c1] by (4, 2) : index
// CHECK: %[[ETILE_VALID_BOUND:.+]] = arith.addi %[[ETILE_VALID]], %c1 : index
// CHECK: %[[ETILE_INVALID:.+]] = affine.linearize_index [%[[PACKED_LASTIDX]]#1, %c0] by (4, 2) : index
// CHECK: %[[ETILE_INVALID_BOUND:.+]] = arith.addi %[[ETILE_INVALID]], %c1 : index
// CHECK: %[[DISTR_LASTIDX:.+]] = affine.linearize_index [%[[PACKED_LASTIDX]]#1, %[[PACKED_LASTIDX]]#3] by (4, 2) : index
// CHECK: %[[DISTR_BOUND:.+]] = arith.addi %[[DISTR_LASTIDX]], %c1 : index

// CHECK: %[[EQ_BOUND_TID:.+]] = arith.cmpi eq, %[[VTID]], %[[PACKED_LASTIDX]]#2 : index
// CHECK: %[[LT_BOUND_TID:.+]] = arith.cmpi slt, %[[VTID]], %[[PACKED_LASTIDX]]#2 : index
// CHECK: %[[EQ_BOUND_SID:.+]] = arith.cmpi eq, %[[VSID]], %[[PACKED_LASTIDX]]#0 : index
// CHECK: %[[LT_BOUND_SID:.+]] = arith.cmpi slt, %[[VSID]], %[[PACKED_LASTIDX]]#0 : index

// CHECK: %[[SELTREE0:.+]] = arith.select %[[LT_BOUND_TID]], %[[ETILE_VALID_BOUND]], %[[ETILE_INVALID_BOUND]] : index
// CHECK: %[[SELTREE1:.+]] = arith.select %[[EQ_BOUND_TID]], %[[DISTR_BOUND]], %[[SELTREE0]] : index
// CHECK: %[[SELTREE2:.+]] = arith.select %[[LT_BOUND_SID]], %c7, %c0 : index
// CHECK: %[[SELTREE3:.+]] = arith.select %[[EQ_BOUND_SID]], %[[SELTREE1]], %[[SELTREE2]] : index
// CHECK: %[[MASK:.+]] = vector.create_mask %[[SELTREE3]], %c7 : vector<8x8xi1>

// CHECK: %[[MASK_EXTR:.+]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 0], sizes = [2, 8], strides = [1, 1]} : vector<8x8xi1> to vector<2x8xi1>
// CHECK: %[[READ:.+]] = vector.transfer_read %arg0{{.*}}, %[[MASK_EXTR]] {in_bounds = [true, true]} : memref<?x128xf16>, vector<2x8xf16>
// CHECK: vector.transfer_write %[[READ]], %arg1{{.*}}, %[[MASK_EXTR]] {in_bounds = [true, true]} : vector<2x8xf16>, memref<?x128xf16>
