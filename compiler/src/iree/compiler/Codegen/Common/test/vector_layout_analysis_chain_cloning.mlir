// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-test-vector-layout-analysis))" --split-input-file %s | FileCheck %s

#layoutA = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layoutB = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [8, 32],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// CHECK-LABEL: @clone_mask_chain
//       CHECK: %[[STEP_A:.+]] = vector.step
//       CHECK: %[[LIMIT_A:.+]] = vector.broadcast %{{.+}} : index to vector<64xindex>
//       CHECK: %[[CMPI_A:.+]] = arith.cmpi slt, %[[STEP_A]], %[[LIMIT_A]]
//       CHECK: %[[MASK_A:.+]] = vector.broadcast %[[CMPI_A]] : vector<64xi1> to vector<16x64xi1>
//       CHECK: %[[STEP_B:.+]] = vector.step
//       CHECK: %[[LIMIT_B:.+]] = vector.broadcast %{{.+}} : index to vector<64xindex>
//       CHECK: %[[CMPI_B:.+]] = arith.cmpi slt, %[[STEP_B]], %[[LIMIT_B]]
//       CHECK: %[[MASK_B:.+]] = vector.broadcast %[[CMPI_B]] : vector<64xi1> to vector<16x64xi1>
//   CHECK-NOT: iree_vector_ext.to_layout {{.*}}xi1
//       CHECK: arith.select %[[MASK_A]]
//       CHECK: arith.select %[[MASK_B]]
func.func @clone_mask_chain(%a: vector<16x64xf16>, %b: vector<16x64xf16>, %n: index) -> (vector<16x64xf16>, vector<16x64xf16>) {
  %cst = arith.constant dense<0.0> : vector<16x64xf16>
  %step = vector.step : vector<64xindex>
  %limit = vector.broadcast %n : index to vector<64xindex>
  %mask_1d = arith.cmpi slt, %step, %limit : vector<64xindex>
  %mask = vector.broadcast %mask_1d : vector<64xi1> to vector<16x64xi1>
  %al = iree_vector_ext.to_layout %a to layout(#layoutA) : vector<16x64xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layoutB) : vector<16x64xf16>
  %sa = arith.select %mask, %al, %cst : vector<16x64xi1>, vector<16x64xf16>
  %sb = arith.select %mask, %bl, %cst : vector<16x64xi1>, vector<16x64xf16>
  func.return %sa, %sb : vector<16x64xf16>, vector<16x64xf16>
}

// -----

#layoutC = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1],
  batch_tile = [1, 8, 1],
  outer_tile = [1, 1, 1],
  thread_tile = [1, 8, 8],
  element_tile = [1, 1, 8],

  subgroup_strides = [0, 0, 0],
  thread_strides   = [0, 8, 1]
>

#layoutD = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1],
  batch_tile = [1, 1, 4],
  outer_tile = [1, 1, 1],
  thread_tile = [1, 4, 16],
  element_tile = [1, 4, 1],

  subgroup_strides = [0, 0, 0],
  thread_strides   = [0, 16, 1]
>

// CHECK-LABEL: @clone_mask_chain_shared_intermediate
//       CHECK: %[[CMPI_A:.+]] = arith.cmpi
//       CHECK: %[[CMPI_B:.+]] = arith.cmpi
//       CHECK: %[[MASK_A:.+]] = vector.broadcast %[[CMPI_A]] : vector<64xi1> to vector<1x64x64xi1>
//       CHECK: %[[MASK_B:.+]] = vector.broadcast %[[CMPI_B]] : vector<64xi1> to vector<1x16x64xi1>
//   CHECK-NOT: iree_vector_ext.to_layout {{.*}}xi1
//       CHECK: arith.select %[[MASK_A]]
//       CHECK: arith.select %[[MASK_B]]
func.func @clone_mask_chain_shared_intermediate(
    %a: vector<1x64x64xf16>, %b: vector<1x16x64xf32>, %n: index)
    -> (vector<1x64x64xf16>, vector<1x16x64xf32>) {
  %cst_f16 = arith.constant dense<0.0> : vector<1x64x64xf16>
  %cst_f32 = arith.constant dense<0.0> : vector<1x16x64xf32>
  %step = vector.step : vector<64xindex>
  %limit = vector.broadcast %n : index to vector<64xindex>
  %mask_1d = arith.cmpi slt, %step, %limit : vector<64xindex>
  %mask_big = vector.broadcast %mask_1d : vector<64xi1> to vector<1x64x64xi1>
  %mask_small = vector.broadcast %mask_1d : vector<64xi1> to vector<1x16x64xi1>
  %al = iree_vector_ext.to_layout %a to layout(#layoutC) : vector<1x64x64xf16>
  %bl = iree_vector_ext.to_layout %b to layout(#layoutD) : vector<1x16x64xf32>
  %sa = arith.select %mask_big, %al, %cst_f16 : vector<1x64x64xi1>, vector<1x64x64xf16>
  %sb = arith.select %mask_small, %bl, %cst_f32 : vector<1x16x64xi1>, vector<1x16x64xf32>
  func.return %sa, %sb : vector<1x64x64xf16>, vector<1x16x64xf32>
}

// -----

// Negative test, chain cloning must stop when the chain reaches a non-cheap op
// (vector.contract in this case). The analysis should insert a `to_layout`
// conversion instead.

#layoutE = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layoutF = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [8, 32],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// CHECK-LABEL: @no_clone_non_cheap_producer
//       CHECK:   %[[NEG:.+]] = arith.negf
//       CHECK:   %[[CONV:.+]] = iree_vector_ext.to_layout %[[NEG]] to layout({{.+}})
//       CHECK:   %[[A:.+]] = iree_vector_ext.to_layout %[[CONV]] to layout({{.+}})
//       CHECK:   %[[B:.+]] = iree_vector_ext.to_layout %[[NEG]] to layout({{.+}})
//       CHECK:   return %[[A]], %[[B]]
func.func @no_clone_non_cheap_producer(
    %lhs: vector<16x32xf16>, %rhs: vector<32x64xf16>, %acc: vector<16x64xf16>)
    -> (vector<16x64xf16>, vector<16x64xf16>) {
  %contract = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<16x32xf16>, vector<32x64xf16> into vector<16x64xf16>
  %neg = arith.negf %contract : vector<16x64xf16>
  %a = iree_vector_ext.to_layout %neg to layout(#layoutE) : vector<16x64xf16>
  %b = iree_vector_ext.to_layout %neg to layout(#layoutF) : vector<16x64xf16>
  func.return %a, %b : vector<16x64xf16>, vector<16x64xf16>
}

// -----

#layoutA = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [16, 64],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

#layoutB = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [2, 2],
  outer_tile = [1, 1],
  thread_tile = [1, 1],
  element_tile = [8, 32],

  subgroup_strides = [0, 0],
  thread_strides   = [0, 0]
>

// Constant-like chain leaves are accepted even if they're defined in another
// block than the rest of the duplicatable chain.

// CHECK-LABEL: @clone_mask_chain_entry_block_constant_leaf
//   CHECK-DAG: %[[CMPI_A:.+]] = arith.cmpi slt
//   CHECK-DAG: %[[CMPI_B:.+]] = arith.cmpi slt
//   CHECK-DAG: %[[MASK_A:.+]] = vector.broadcast %[[CMPI_A]] : vector<64xi1> to vector<16x64xi1>
//   CHECK-DAG: %[[MASK_B:.+]] = vector.broadcast %[[CMPI_B]] : vector<64xi1> to vector<16x64xi1>
//   CHECK-NOT: iree_vector_ext.to_layout {{.*}}xi1
//       CHECK: arith.select %[[MASK_A]]
//       CHECK: arith.select %[[MASK_B]]
func.func @clone_mask_chain_entry_block_constant_leaf(
    %a: vector<16x64xf16>, %b: vector<16x64xf16>, %cond: i1)
    -> (vector<16x64xf16>, vector<16x64xf16>) {
  %cst = arith.constant dense<24> : vector<64xindex>
  %zero = arith.constant dense<0.0> : vector<16x64xf16>
  %0:2 = scf.if %cond -> (vector<16x64xf16>, vector<16x64xf16>) {
    %step = vector.step : vector<64xindex>
    %mask_1d = arith.cmpi slt, %step, %cst : vector<64xindex>
    %mask = vector.broadcast %mask_1d : vector<64xi1> to vector<16x64xi1>
    %al = iree_vector_ext.to_layout %a to layout(#layoutA) : vector<16x64xf16>
    %bl = iree_vector_ext.to_layout %b to layout(#layoutB) : vector<16x64xf16>
    %sa = arith.select %mask, %al, %zero : vector<16x64xi1>, vector<16x64xf16>
    %sb = arith.select %mask, %bl, %zero : vector<16x64xi1>, vector<16x64xf16>
    scf.yield %sa, %sb : vector<16x64xf16>, vector<16x64xf16>
  } else {
    scf.yield %zero, %zero : vector<16x64xf16>, vector<16x64xf16>
  }
  func.return %0#0, %0#1 : vector<16x64xf16>, vector<16x64xf16>
}
