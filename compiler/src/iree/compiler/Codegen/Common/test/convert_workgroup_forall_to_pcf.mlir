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

//   CHECK-DAG:   %[[COUNTY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBY]], %[[LBY]], %[[STEPY]]]
//   CHECK-DAG:   %[[COUNTX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBX]], %[[LBX]], %[[STEPX]]]
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup) count(%[[COUNTX]], %[[COUNTY]])
//  CHECK-NEXT:       execute[%[[IDX:[A-Za-z0-9_]+]]: index, %[[IDY:[A-Za-z0-9_]+]]: index]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[IDY]], %[[LBY]], %[[STEPY]]]
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[IDX]], %[[LBX]], %[[STEPX]]]
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
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup) count(%[[COUNTX]], %[[COUNTY]])
//  CHECK-NEXT:       execute[%[[IDX:[A-Za-z0-9_]+]]: index, %[[IDY:[A-Za-z0-9_]+]]: index]
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[IDX]], %[[LBX]], %[[STEPX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[IDY]], %[[LBY]], %[[STEPY]]]
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

//   CHECK-DAG:   %[[COUNTZ0:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//   CHECK-DAG:   %[[COUNTZ1:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) ceildiv s2)>()[%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]]]
//       CHECK:   %[[COUNTZ:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5]
//  CHECK-SAME:    -> (((s0 - s1) ceildiv s2) * ((s3 - s4) ceildiv s5))>()
//  CHECK-SAME:    [%[[UBZ1]], %[[LBZ1]], %[[STEPZ1]], %[[UBZ0]], %[[LBZ0]], %[[STEPZ0]]]
//       CHECK:   pcf.loop scope(#iree_codegen.workgroup) count(%c6, %c2, %[[COUNTZ]])
//  CHECK-NEXT:       execute[%[[IDX:[A-Za-z0-9_]+]]: index, %[[IDY:[A-Za-z0-9_]+]]: index, %[[IDZ:[A-Za-z0-9_]+]]: index]
//       CHECK:     %[[DELIN:.+]]:2 = affine.delinearize_index %[[IDZ]] into (%[[COUNTZ1]], %[[COUNTZ0]]) : index, index
//   CHECK-DAG:     %[[CIDX:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 + 5)>()[%[[IDX]]]
//   CHECK-DAG:     %[[CIDY:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0] -> (s0 * 2 + 6)>()[%[[IDY]]]
//   CHECK-DAG:     %[[CIDZ0:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#1, %[[LBZ0]], %[[STEPZ0]]]
//   CHECK-DAG:     %[[CIDZ1:[A-Za-z0-9_]+]] = affine.apply affine_map<()[s0, s1, s2] -> ((s0 + s1) * s2)>()[%[[DELIN]]#0, %[[LBZ1]], %[[STEPZ1]]]
//       CHECK:     "use"(%[[CIDY]], %[[CIDZ0]], %[[CIDX]], %[[CIDZ1]])