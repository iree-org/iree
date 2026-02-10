// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-convert-workgroup-forall-to-pcf)" \
// RUN:  --allow-unregistered-dialect --mlir-print-local-scope --split-input-file | FileCheck %s

func.func @convert_forall(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index) {
  scf.forall (%arg0, %arg1) = (%0, %1) to (%2, %3) step(%4, %5) {
    "use"(%arg0, %arg1) : (index, index) -> ()
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  return
}

// CHECK-LABEL: @convert_forall
//  CHECK-SAME:   %[[LBY:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[LBX:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBY:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBX:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPY:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPX:[A-Za-z0-9_]+]]: index

//   CHECK-DAG:   %[[COUNTX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBX]], %[[LBX]], %[[STEPX]]]
//   CHECK-DAG:   %[[COUNTY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBY]], %[[LBY]], %[[STEPY]]]
//   CHECK-DAG:   %[[LINEARIZED_COUNT:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> (((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5))>()[%[[UBY]], %[[LBY]], %[[STEPY]], %[[UBX]], %[[LBX]], %[[STEPX]]]
//       CHECK:   iree_codegen.workgroup_count_hint(%[[LINEARIZED_COUNT]])
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:2 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTY]], %[[COUNTX]]) : index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#1, %[[STEPX]], %[[LBX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#0, %[[STEPY]], %[[LBY]]]
//       CHECK:     "use"(%[[CIDY]], %[[CIDX]])

// -----

func.func @convert_forall_reverse_mapping(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index) {
  scf.forall (%arg0, %arg1) = (%0, %1) to (%2, %3) step(%4, %5) {
    "use"(%arg0, %arg1) : (index, index) -> ()
  } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}
  return
}

// CHECK-LABEL: @convert_forall_reverse_mapping
//  CHECK-SAME:   %[[LBX:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[LBY:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBX:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBY:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPX:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPY:[A-Za-z0-9_]+]]: index

//   CHECK-DAG:   %[[COUNTX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBX]], %[[LBX]], %[[STEPX]]]
//   CHECK-DAG:   %[[COUNTY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBY]], %[[LBY]], %[[STEPY]]]
//   CHECK-DAG:   %[[LINEARIZED_COUNT:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> (((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5))>()[%[[UBY]], %[[LBY]], %[[STEPY]], %[[UBX]], %[[LBX]], %[[STEPX]]]
//       CHECK:   iree_codegen.workgroup_count_hint(%[[LINEARIZED_COUNT]])
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:2 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTY]], %[[COUNTX]]) : index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#1, %[[STEPX]], %[[LBX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#0, %[[STEPY]], %[[LBY]]]
//       CHECK:     "use"(%[[CIDX]], %[[CIDY]])

// -----

func.func @convert_forall_delinearize(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index) {
  scf.forall (%arg0, %arg1, %arg2, %arg3) = (3, %0, 5, %1) to (7, %2, 11, %3) step(2, %4, 1, %5) {
    "use"(%arg0, %arg1, %arg2, %arg3) : (index, index, index, index) -> ()
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<z:0>, #iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<z:1>]}
  return
}

// CHECK-LABEL: @convert_forall_delinearize
//  CHECK-SAME:   %[[LBZ0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[LBZ1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBZ0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[UBZ1:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPZ0:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[STEPZ1:[A-Za-z0-9_]+]]: index

//   CHECK-DAG:   %[[C6:[a-zA-Z0-9_]+]] = arith.constant 6
//   CHECK-DAG:   %[[C2:[a-zA-Z0-9_]+]] = arith.constant 2
//   CHECK-DAG:   %[[COUNTZ0:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//   CHECK-DAG:   %[[COUNTZ1:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]]]
//   CHECK-DAG:   %[[COUNTZ:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> (((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5))>()[%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]], %[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//       CHECK:   %{{.+}} = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> ((((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5)) * 2)>()[%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]], %[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//       CHECK:   %[[LINEARIZED_COUNT:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> ((((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5)) * 12)>()[%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]], %[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//       CHECK:   iree_codegen.workgroup_count_hint(%[[LINEARIZED_COUNT]])
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:4 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTZ1]], %[[COUNTZ0]], 2, 6) : index, index, index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%[[DELIN]]#3]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 * 2 + 3)>()[%[[DELIN]]#2]
//   CHECK-DAG:     %[[CIDZ0:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#1, %[[STEPZ0]], %[[LBZ0]]]
//   CHECK-DAG:     %[[CIDZ1:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELIN]]#0, %[[STEPZ1]], %[[LBZ1]]]
//       CHECK:     "use"(%[[CIDY]], %[[CIDZ0]], %[[CIDX]], %[[CIDZ1]])

// -----

// Forall without mapping should not be converted.
func.func @forall_no_mapping_not_converted() {
  scf.forall (%arg0, %arg1) = (0, 0) to (32, 32) step (4, 4) {
    "use"(%arg0, %arg1) : (index, index) -> ()
  }
  return
}

// CHECK-LABEL: @forall_no_mapping_not_converted
// CHECK: scf.forall
// CHECK-NOT: pcf.loop

// -----

// Forall with non-workgroup mapping (gpu.thread) should not be converted.
func.func @forall_gpu_thread_mapping_not_converted() {
  scf.forall (%arg0, %arg1) = (0, 0) to (32, 32) step (4, 4) {
    "use"(%arg0, %arg1) : (index, index) -> ()
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  return
}

// CHECK-LABEL: @forall_gpu_thread_mapping_not_converted
// CHECK: scf.forall
// CHECK-NOT: pcf.loop

// -----

// Test folding split-reduction forall containing workgroup pcf.loop into pcf.generic.
func.func @fold_split_reduction_into_pcf_generic(%init: tensor<16xf32>, %slice: tensor<1xf32>) -> tensor<16xf32> {
  %c4 = arith.constant 4 : index
  %0 = scf.forall (%id) in (4) shared_outs(%iter = %init) -> (tensor<16xf32>) {
    %tile_init = tensor.extract_slice %iter[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %loop_result = pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%c4)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<4xf32, sync(#iree_codegen.workgroup_scope<linearize>)>)
           -> (tensor<4xf32>) {
      pcf.write_slice %slice into %ref[%loop_id] [1] [1]
          : tensor<1xf32> into !pcf.sref<4xf32, sync(#iree_codegen.workgroup_scope<linearize>)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
    }
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  return %0 : tensor<16xf32>
}

// Total workgroup count = forall iterations (4) * loop count (4).
// The arith.muli computes the product of the forall upper bound and the loop count.
// CHECK-LABEL: @fold_split_reduction_into_pcf_generic
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<16xf32>
//  CHECK-SAME:   %[[SLICE:[A-Za-z0-9_]+]]: tensor<1xf32>
//   CHECK-DAG:   %[[C4_A:[a-zA-Z0-9_]+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C4_B:[a-zA-Z0-9_]+]] = arith.constant 4 : index
//       CHECK:   %[[TOTAL:.+]] = arith.muli %[[C4_B]], %[[C4_A]] : index
//       CHECK:   iree_codegen.workgroup_count_hint(%[[TOTAL]])
//       CHECK:   %[[GENERIC:.+]] = pcf.generic
//       CHECK:     scope(#iree_codegen.workgroup_scope<linearize>)
//       CHECK:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[INIT]])[%[[GEN_ID:[A-Za-z0-9_]+]]: index, %{{.*}}: index]
//       CHECK:          : (!pcf.sref<16xf32, sync(#iree_codegen.workgroup_scope<linearize>)>)
//       CHECK:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[GEN_ID]] into
//       CHECK:     scf.forall (%{{.+}}) = (%[[DELIN]]#0)
//       CHECK:       scf.forall (%{{.+}}) = (%[[DELIN]]#1)
//       CHECK:         pcf.write_slice %[[SLICE]] into %[[REF]]{{.*}} [1] [1]
//  CHECK-SAME:           into !pcf.sref<16xf32, sync(#iree_codegen.workgroup_scope<linearize>)>
//       CHECK:     pcf.return
//       CHECK:   return %[[GENERIC]]

// -----

// Non-split-reduction mapping should not be folded by the split-k pattern.
// The workgroup forall is converted to pcf.loop, but the outer forall (with
// local mapping) should remain unconverted.
func.func @non_split_reduction_not_folded(%init: tensor<16xf32>, %slice: tensor<1xf32>) -> tensor<16xf32> {
  %c4 = arith.constant 4 : index
  %0 = scf.forall (%id) in (4) shared_outs(%iter = %init) -> (tensor<16xf32>) {
    %tile_init = tensor.extract_slice %iter[%id] [4] [1]
        : tensor<16xf32> to tensor<4xf32>
    %loop_result = pcf.loop scope(#iree_codegen.workgroup_scope<linearize>) count(%c4)
        execute(%ref = %tile_init)[%loop_id: index]
            : (!pcf.sref<4xf32, sync(#iree_codegen.workgroup_scope<linearize>)>)
           -> (tensor<4xf32>) {
      pcf.write_slice %slice into %ref[%loop_id] [1] [1]
          : tensor<1xf32> into !pcf.sref<4xf32, sync(#iree_codegen.workgroup_scope<linearize>)>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %loop_result into %iter[%id] [4] [1]
          : tensor<4xf32> into tensor<16xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: @non_split_reduction_not_folded
//       CHECK:   scf.forall
//       CHECK:     pcf.loop
//   CHECK-NOT:   pcf.generic
