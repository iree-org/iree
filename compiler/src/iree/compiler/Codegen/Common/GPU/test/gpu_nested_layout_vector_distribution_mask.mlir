// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse --canonicalize --mlir-print-local-scope %s | FileCheck %s

#nested = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [8, 1],
  outer_tile = [2, 1],
  thread_tile = [4, 16],
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
// CHECK-DAG: %[[DIM:.+]] = memref.dim %arg0, %c0 : memref<?x128xf16>
// CHECK: %[[SID:.+]]:3 = affine.delinearize_index %thread_id_x into (2, 64)
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %thread_id_x into (4, 16)
// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[STEP_BC:.+]] = vector.broadcast %[[STEP]] : vector<2xindex> to vector<16x2xindex>
// CHECK: %[[BASE_IDX:.+]] = arith.addi %{{.+}}, %[[STEP_BC]] : vector<16x2xindex>
// CHECK: %[[SID_OFF:.+]] = arith.muli %[[SID]]#1, %c128 : index
// CHECK: %[[SID_OFF_BC:.+]] = vector.broadcast %[[SID_OFF]] : index to vector<16x2xindex>
// CHECK: %[[IDX0:.+]] = arith.addi %[[BASE_IDX]], %[[SID_OFF_BC]] : vector<16x2xindex>
// CHECK: %[[TID_OFF:.+]] = arith.muli %[[TID]]#1, %c2 : index
// CHECK: %[[TID_OFF_BC:.+]] = vector.broadcast %[[TID_OFF]] : index to vector<16x2xindex>
// CHECK: %[[IDX1:.+]] = arith.addi %[[IDX0]], %[[TID_OFF_BC]] : vector<16x2xindex>
// CHECK: %[[IDX_VEC:.+]] = vector.shape_cast %[[IDX1]] : vector<16x2xindex> to vector<8x2x2xindex>
// CHECK: %[[BOUND_BC:.+]] = vector.broadcast %[[DIM]] : index to vector<8x2x2xindex>
// CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[IDX_VEC]], %[[BOUND_BC]] : vector<8x2x2xindex>
// CHECK: %[[FLAT_MASK:.+]] = vector.shape_cast %{{.+}} : vector<8x1x2x1x2x8xi1> to vector<32x8xi1>
// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice %[[FLAT_MASK]] {offsets = [0, 0], sizes = [2, 8]{{.*}}} : vector<32x8xi1> to vector<2x8xi1>
// CHECK: vector.transfer_read %arg0{{.*}}, %[[SLICE0]] {in_bounds = [true, true]} : memref<?x128xf16>, vector<2x8xf16>
// CHECK: vector.transfer_write {{.*}}, %arg1{{.*}} {in_bounds = [true, true]} : vector<2x8xf16>, memref<?x128xf16>

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

// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice {{.*}} {offsets = [0, 0, 0], sizes = [8, 2, 1]{{.*}}} : vector<8x8x1xi1> to vector<8x2x1xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[SLICE0]] {in_bounds = [true, true, true]{{.*}}} : memref<128x?x1xf16>, vector<1x2x8xf16>

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
// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice {{.*}} {offsets = [0, 0], sizes = [8, 2]{{.*}}} : vector<8x8xi1> to vector<8x2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[SLICE0]] {in_bounds = [true, true]{{.*}}} : memref<128x?x1xf16>, vector<2x8xf16>

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

// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[BOUND_BC:.+]] = vector.broadcast %{{.+}} : index to vector<2x2x2xindex>
// CHECK: %[[CMP:.+]] = arith.cmpi slt, %{{.+}}, %[[BOUND_BC]] : vector<2x2x2xindex>
// CHECK: %[[FLAT_MASK:.+]] = vector.shape_cast %[[CMP]] : vector<2x2x2xi1> to vector<8xi1>
// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice %[[FLAT_MASK]] {offsets = [0], sizes = [2]{{.*}}} : vector<8xi1> to vector<2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[SLICE0]] {in_bounds = [true, true]{{.*}}} : memref<128x?x1xf16>, vector<2x8xf16>

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
// CHECK-DAG: %[[RED_IDENTITY:.+]] = arith.constant dense<0.000000e+00> : vector<2x1x2x1x2x8xf16>

// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[CMP:.+]] = arith.cmpi slt, %{{.+}}, %{{.+}} : vector<2x2x2xindex>
// CHECK: %[[MASK_BCAST:.+]] = vector.broadcast %[[CMP]] : vector<2x2x2xi1> to vector<1x1x8x2x2x2xi1>
// CHECK: %[[MASK_PACKED:.+]] = vector.transpose %[[MASK_BCAST]], [3, 0, 4, 1, 5, 2] : vector<1x1x8x2x2x2xi1> to vector<2x1x2x1x2x8xi1>

// CHECK: %[[SELECT:.+]] = arith.select %[[MASK_PACKED]], %{{.+}}, %[[RED_IDENTITY]] : vector<2x1x2x1x2x8xi1>, vector<2x1x2x1x2x8xf16>
// CHECK: vector.multi_reduction <add>, %[[SELECT]], %{{.+}} [0, 2, 4] : vector<2x1x2x1x2x8xf16> to vector<1x1x8xf16>

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

//     Step+cmpi patterns produce distributed masks for LHS (reddim) and output (pardim).
// CHECK: %[[LHS_CMP:.+]] = arith.cmpi slt, %{{.+}}, %{{.+}} : vector<1x1x2xindex>
// CHECK: %[[OUT_CMP:.+]] = arith.cmpi slt, %{{.+}}, %{{.+}} : vector<1x1x2xindex>

//     Broadcast each 1D mask to the full 2D shape.  Because neither is the
//     trailing dim of its target, each goes through broadcast (permuted) then
//     transpose back.  CSE deduplicates the broadcasts across the two andi's.
// CHECK: %[[OUT_BCAST:.+]] = vector.broadcast %[[OUT_CMP]] : vector<1x1x2xi1> to vector<1x1x1x1x2x2xi1>
// CHECK: %[[OUT_TRANS:.+]] = vector.transpose %[[OUT_BCAST]], [1, 0, 3, 2, 5, 4] : vector<1x1x1x1x2x2xi1> to vector<1x1x1x1x2x2xi1>
// CHECK: %[[LHS_BCAST:.+]] = vector.broadcast %[[LHS_CMP]] : vector<1x1x2xi1> to vector<1x1x1x1x2x2xi1>

//     RHS read mask = pardim (transposed back) & reddim (broadcast).
// CHECK: %[[RHS_READ_MASK:.+]] = arith.andi %[[OUT_TRANS]], %[[LHS_BCAST]] : vector<1x1x1x1x2x2xi1>

//     Contraction mask = reddim (transposed back) & pardim (broadcast).
//     Reuses the CSE'd broadcasts (LHS_BCAST, OUT_BCAST).
// CHECK: %[[LHS_TRANS:.+]] = vector.transpose %[[LHS_BCAST]], [1, 0, 3, 2, 5, 4] : vector<1x1x1x1x2x2xi1> to vector<1x1x1x1x2x2xi1>
// CHECK: %[[CONTRACT_MASK:.+]] = arith.andi %[[LHS_TRANS]], %[[OUT_BCAST]] : vector<1x1x1x1x2x2xi1>

//     LHS read uses the reddim mask directly.
// CHECK: %[[LHS_MASK_FLAT:.+]] = vector.shape_cast %[[LHS_CMP]] : vector<1x1x2xi1> to vector<2xi1>
// CHECK: vector.transfer_read %arg0{{.*}} %[[LHS_MASK_FLAT]] {in_bounds = [true]} : memref<?xf16>, vector<2xf16>

//     RHS read uses the RHS read mask (pardim anded with reddim).
// CHECK: %[[RHS_READ_MASK_FLAT:.+]] = vector.shape_cast %[[RHS_READ_MASK]] : vector<1x1x1x1x2x2xi1> to vector<2x2xi1>
// CHECK: vector.transfer_read %arg1{{.*}} %[[RHS_READ_MASK_FLAT]] {in_bounds = [true, true]} : memref<?x?xf16>, vector<2x2xf16>

//     Contraction mask is flattened, transposed, and split into per-operand
//     select masks for the reduction identity replacement.
// CHECK: %[[CM_FLAT:.+]] = vector.shape_cast %[[CONTRACT_MASK]] : vector<1x1x1x1x2x2xi1> to vector<2x2xi1>
// CHECK: %[[CM_TRANS:.+]] = vector.transpose %[[CM_FLAT]], [1, 0] : vector<2x2xi1> to vector<2x2xi1>
// CHECK: %[[LHS_SEL_MASK:.+]] = vector.extract %[[CM_TRANS]][0] : vector<2xi1> from vector<2x2xi1>
// CHECK: %[[LHS_SEL_MASK_3D:.+]] = vector.shape_cast %[[LHS_SEL_MASK]] : vector<2xi1> to vector<1x1x2xi1>
// CHECK: %[[RHS_SEL_MASK:.+]] = vector.shape_cast %[[CM_TRANS]] : vector<2x2xi1> to vector<1x1x1x1x2x2xi1>

//     Select with reduction identity, then contract.
// CHECK: %[[LHS_SELECT:.+]] = arith.select %[[LHS_SEL_MASK_3D]], %{{.+}}, %[[RED_IDENTITY_LHS]] : vector<1x1x2xi1>, vector<1x1x2xf16>
// CHECK: %[[RHS_SELECT:.+]] = arith.select %[[RHS_SEL_MASK]], %{{.+}}, %[[RED_IDENTITY_RHS]] : vector<1x1x1x1x2x2xi1>, vector<1x1x1x1x2x2xf16>
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
// CHECK-DAG: %[[BOUND:.+]] = arith.constant dense<17> : vector<2x2x2xindex>
// CHECK: %[[SID:.+]]:3 = affine.delinearize_index %thread_id_x into (2, 64)
// CHECK: %[[TID:.+]]:3 = affine.delinearize_index %thread_id_x into (16, 16)
// CHECK: %[[STEP:.+]] = vector.step : vector<2xindex>
// CHECK: %[[STEP_BC:.+]] = vector.broadcast %[[STEP]] : vector<2xindex> to vector<4x2xindex>
// CHECK: %[[BASE_IDX:.+]] = arith.addi %{{.+}}, %[[STEP_BC]] : vector<4x2xindex>
// CHECK: %[[SID_OFF:.+]] = arith.muli %[[SID]]#1, %c128 : index
// CHECK: %[[SID_OFF_BC:.+]] = vector.broadcast %[[SID_OFF]] : index to vector<4x2xindex>
// CHECK: %[[IDX0:.+]] = arith.addi %[[BASE_IDX]], %[[SID_OFF_BC]] : vector<4x2xindex>
// CHECK: %[[TID_OFF:.+]] = arith.muli %[[TID]]#1, %c2 : index
// CHECK: %[[TID_OFF_BC:.+]] = vector.broadcast %[[TID_OFF]] : index to vector<4x2xindex>
// CHECK: %[[IDX1:.+]] = arith.addi %[[IDX0]], %[[TID_OFF_BC]] : vector<4x2xindex>
// CHECK: %[[IDX_VEC:.+]] = vector.shape_cast %[[IDX1]] : vector<4x2xindex> to vector<2x2x2xindex>
// CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[IDX_VEC]], %[[BOUND]] : vector<2x2x2xindex>

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
  %mask = vector.create_mask %c7, %c7 : vector<16x8xi1>

  %out = iree_vector_ext.transfer_gather %source[%c0, %c0, %c0]
  [%indices : vector<16xindex>], %cst0, %mask {
    indexing_maps = [affine_map<(d0, d1)[s0] -> (0, s0, d1)>,
                     affine_map<(d0, d1)[s0] -> (d0)>,
                     affine_map<(d0, d1)[s0] -> (d0, d1)>]
  } : memref<4096x512x8xf16>, vector<16x8xf16>, vector<16x8xi1>

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
// CHECK-DAG: %[[ROW_BOUND:.+]] = arith.constant dense<7> : vector<4x1x1xindex>
// CHECK-DAG: %[[COL_BOUND:.+]] = arith.constant dense<7> : vector<1x1x8xindex>
// CHECK: arith.cmpi slt, %{{.+}}, %[[ROW_BOUND]] : vector<4x1x1xindex>
// CHECK: arith.cmpi slt, %{{.+}}, %[[COL_BOUND]] : vector<1x1x8xindex>
// CHECK: %[[MASK_2D:.+]] = arith.andi %{{.+}}, %{{.+}} : vector<4x1x1x1x1x8xi1>
// CHECK: %[[M0:.+]] = vector.extract %[[MASK_2D]][0, 0, 0, 0, 0] : vector<8xi1>
// CHECK: vector.transfer_read %arg1{{.*}} %[[M0]] {in_bounds = [true, true]{{.*}}} : memref<4096x512x8xf16>, vector<1x8xf16>
// CHECK: %[[M1:.+]] = vector.extract %[[MASK_2D]][1, 0, 0, 0, 0] : vector<8xi1>
// CHECK: vector.transfer_read %arg1{{.*}} %[[M1]] {in_bounds = [true, true]{{.*}}} : memref<4096x512x8xf16>, vector<1x8xf16>
// CHECK: %[[M2:.+]] = vector.extract %[[MASK_2D]][2, 0, 0, 0, 0] : vector<8xi1>
// CHECK: vector.transfer_read %arg1{{.*}} %[[M2]] {in_bounds = [true, true]{{.*}}} : memref<4096x512x8xf16>, vector<1x8xf16>
// CHECK: %[[M3:.+]] = vector.extract %[[MASK_2D]][3, 0, 0, 0, 0] : vector<8xi1>
// CHECK: vector.transfer_read %arg1{{.*}} %[[M3]] {in_bounds = [true, true]{{.*}}} : memref<4096x512x8xf16>, vector<1x8xf16>

// -----

// This test covers the case constant_mask [2, 508] on a 2x512
// vector with element_tile=1 and batch_tile=8 on the masked dimension.
// Last valid index 507 delinearizes to (u1=7, u3=59). Threads 60-63 (tid > 59)
// must get postThreadsBound=7 (not 0), because they still have 7 valid batch
// iterations before the boundary batch.

#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 8],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],

  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

func.func @masked_read_write_unit_element_tile(%arg0 : memref<2x512xf16>, %arg1 : memref<2x512xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %mask = vector.constant_mask [2, 508] : vector<2x512xi1>
  %read = vector.transfer_read %arg0[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : memref<2x512xf16>, vector<2x512xf16>
  %layout = iree_vector_ext.to_layout %read to layout(#layout) : vector<2x512xf16>
  vector.transfer_write %layout, %arg1[%c0, %c0], %mask {in_bounds = [true, true]} : vector<2x512xf16>, memref<2x512xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_unit_element_tile
// CHECK-DAG: %[[BOUND:.+]] = arith.constant dense<508> : vector<8x1x1xindex>
// CHECK: arith.cmpi slt, %{{.+}}, %[[BOUND]] : vector<8x1x1xindex>
// CHECK: %[[FLAT_MASK:.+]] = vector.shape_cast %{{.+}} : vector<2x1x1x8x1x1xi1> to vector<2x8xi1>
// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice %[[FLAT_MASK]] {offsets = [0, 0], sizes = [1, 1]{{.*}}} : vector<2x8xi1> to vector<1x1xi1>
// CHECK: vector.transfer_read %arg0{{.*}}, %[[SLICE0]] {in_bounds = [true, true]} : memref<2x512xf16>, vector<1x1xf16>

// -----

// This test exercises the three-way thread split when element_tile > 1. With
// constant_mask [58] on a 64-element vector (batch=4, thread=4, element=4),
// last valid index 57 delinearizes to (u1=3, u3=2, u4=1). All three cases
// produce distinct bounds:
//   tid < 2:  16 (4 full batches)
//   tid == 2: 14 (3 full batches + partial element tile of 2)
//   tid > 2:  12 (3 full batches, boundary batch has 0 valid elements)

#layout_1d = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [4],
  outer_tile = [1],
  thread_tile = [4],
  element_tile = [4],

  subgroup_strides = [0],
  thread_strides = [1]
>

func.func @masked_read_write_partial_element_tile(%arg0 : memref<64xf16>, %arg1 : memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %mask = vector.constant_mask [58] : vector<64xi1>
  %read = vector.transfer_read %arg0[%c0], %cst, %mask {in_bounds = [true]} : memref<64xf16>, vector<64xf16>
  %layout = iree_vector_ext.to_layout %read to layout(#layout_1d) : vector<64xf16>
  vector.transfer_write %layout, %arg1[%c0], %mask {in_bounds = [true]} : vector<64xf16>, memref<64xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @masked_read_write_partial_element_tile
// CHECK-DAG: %[[BOUND:.+]] = arith.constant dense<58> : vector<4x1x4xindex>
// CHECK: %[[CMP:.+]] = arith.cmpi slt, %{{.+}}, %[[BOUND]] : vector<4x1x4xindex>
// CHECK: %[[FLAT_MASK:.+]] = vector.shape_cast %[[CMP]] : vector<4x1x4xi1> to vector<16xi1>
// CHECK: %[[SLICE0:.+]] = vector.extract_strided_slice %[[FLAT_MASK]] {offsets = [0], sizes = [4]{{.*}}} : vector<16xi1> to vector<4xi1>
// CHECK: vector.transfer_read %arg0{{.*}}, %[[SLICE0]] {in_bounds = [true]} : memref<64xf16>, vector<4xf16>
