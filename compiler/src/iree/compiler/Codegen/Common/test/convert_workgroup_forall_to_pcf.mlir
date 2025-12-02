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
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:2 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTY]], %[[COUNTX]]) : index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#1, %[[LBX]], %[[STEPX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#0, %[[LBY]], %[[STEPY]]]
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
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:2 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTY]], %[[COUNTX]]) : index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#1, %[[LBX]], %[[STEPX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#0, %[[LBY]], %[[STEPY]]]
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
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup<linearize>) count(%[[LINEARIZED_COUNT]])
//  CHECK-NEXT:       execute[%[[LINEAR_ID:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:[A-Za-z0-9_]+]]:4 = affine.delinearize_index %[[LINEAR_ID]] into (%[[COUNTZ1]], %[[COUNTZ0]], 2, 6) : index, index, index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%[[DELIN]]#3]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 * 2 + 6)>()[%[[DELIN]]#2]
//   CHECK-DAG:     %[[CIDZ0:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#1, %[[LBZ0]], %[[STEPZ0]]]
//   CHECK-DAG:     %[[CIDZ1:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#0, %[[LBZ1]], %[[STEPZ1]]]
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
