// RUN: iree-opt --iree-codegen-fold-split-reduction-and-workgroup-mapping-loops --split-input-file --mlir-print-local-scope --allow-unregistered-dialect %s | FileCheck %s

func.func @simple_example_1dmapping(%0 : index, %1 : index, %2 : index, %3 : index,
    %4 : index, %5 : index) {
  scf.forall (%arg0) = (%0) to (%1) step (%2) {
    "use1"(%arg0) : (index) -> ()
    scf.forall (%arg1) = (%3) to (%4) step (%5) {
      "use2"(%arg0, %arg1) : (index, index) -> ()
    } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  return
}
//      CHECK: func @simple_example_1dmapping
// CHECK-SAME:     %[[SPLIT_LB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_UB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_STEP:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[WG_LB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[WG_UB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[WG_STEP:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.forall
// CHECK-SAME:       %[[IV0:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV1:[a-zA-Z0-9]+]]
// CHECK-SAME:       = (%[[SPLIT_LB]], %[[WG_LB]])
// CHECK-SAME:       to (%[[SPLIT_UB]], %[[WG_UB]])
// CHECK-SAME:       step (%[[SPLIT_STEP]], %[[WG_STEP]])
//      CHECK:     "use1"(%[[IV0]])
//      CHECK:     "use2"(%[[IV0]], %[[IV1]])
//      CHECK:     mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

func.func @simple_example_2dmapping(%0 : index, %1 : index, %2 : index, %3 : index,
    %4 : index) {
  scf.forall (%arg0) = (%0) to (%1) step (%2) {
    "use1"(%arg0) : (index) -> ()
    scf.forall (%arg1, %arg2)  in (%3, %4) {
      "use2"(%arg0, %arg1, %arg2) : (index, index, index) -> ()
    } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  return
}
//      CHECK: func @simple_example_2dmapping
// CHECK-SAME:     %[[SPLIT_LB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_UB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_STEP:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[WG_UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[WG_UB1:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.forall
// CHECK-SAME:       %[[IV0:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV1:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV2:[a-zA-Z0-9]+]]
// CHECK-SAME:       = (%[[SPLIT_LB]], 0, 0)
// CHECK-SAME:       to (%[[SPLIT_UB]], %[[WG_UB0]], %[[WG_UB1]])
// CHECK-SAME:       step (%[[SPLIT_STEP]], 1, 1)
//      CHECK:     "use1"(%[[IV0]])
//      CHECK:     "use2"(%[[IV0]], %[[IV1]], %[[IV2]])
//      CHECK:     mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

func.func @simple_example_3dmapping(%0 : index, %1 : index, %2 : index, %3 : index,
    %4 : index, %5 : index) {
  scf.forall (%arg0) = (%0) to (%1) step (%2) {
    "use1"(%arg0) : (index) -> ()
    scf.forall (%arg1, %arg2, %arg3) in (%3, %4, %5) {
      "use2"(%arg1, %arg2, %arg3) : (index, index, index) -> ()
    } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  return
}
//      CHECK: func @simple_example_3dmapping
// CHECK-SAME:     %[[SPLIT_LB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_UB:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[SPLIT_STEP:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ORIG_UB0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ORIG_UB1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[ORIG_UB2:[a-zA-Z0-9_]+]]: index
//      CHECK:   scf.forall
// CHECK-SAME:       %[[IV0:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV1:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV2:[a-zA-Z0-9]+]]
// CHECK-SAME:       %[[IV3:[a-zA-Z0-9]+]]
// CHECK-SAME:       = (%[[SPLIT_LB]], 0, 0, 0)
// CHECK-SAME:       to (%[[SPLIT_UB]], %[[ORIG_UB0]], %[[ORIG_UB1]], %[[ORIG_UB2]])
// CHECK-SAME:       step (%[[SPLIT_STEP]], 1, 1, 1)
//      CHECK:     "use1"(%[[IV0]])
//      CHECK:     "use2"(%[[IV1]], %[[IV2]], %[[IV3]])
//      CHECK:     mapping = [#iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]

// -----

// Resolution when the split reduction `forall` and workgroup mapping
// `forall` are multi-dimensional.

func.func @split_2d_3dmapping(%0 : index, %1 : index, %2 : index, %3 : index,
    %4 : index) {
  scf.forall (%arg0, %arg1) in (%0, %1) {
    "use1"(%arg0, %arg1) : (index, index) -> ()
    scf.forall (%arg2, %arg3, %arg4) in (%2, %3, %4) {
      "use2"(%arg2, %arg3, %arg4) : (index, index, index) -> ()
    } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<1>, #iree_linalg_ext.split_reduction_mapping<0>]}
  return
}
// CHECK-LABEL: @split_2d_3dmapping(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: index)
//       CHECK:   scf.forall (
//  CHECK-SAME:       %[[IV0:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV1:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV2:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV3:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV4:[a-zA-Z0-9]+]])
//  CHECK-SAME:       in (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]])
//   CHECK-DAG:     "use1"(%[[IV0]], %[[IV1]])
//   CHECK-DAG:     "use2"(%[[IV2]], %[[IV3]], %[[IV4]])
//       CHECK:     mapping = [
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z:2>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z:1>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<y>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<x>]

// -----


// Resolution when the split reduction `forall` and workgroup mapping
// `forall` are multi-dimensional and the mappings are permuted

func.func @split_2d_3dmapping_permuted(%0 : index, %1 : index, %2 : index, %3 : index,
    %4 : index) {
  scf.forall (%arg0, %arg1) in (%0, %1) {
    "use1"(%arg0, %arg1) : (index, index) -> ()
    scf.forall (%arg2, %arg3, %arg4) in (%2, %3, %4) {
      "use2"(%arg2, %arg3, %arg4) : (index, index, index) -> ()
    } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<z>]}
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>, #iree_linalg_ext.split_reduction_mapping<1>]}
  return
}
// CHECK-LABEL: @split_2d_3dmapping_permuted(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: index)
//       CHECK:   scf.forall (
//  CHECK-SAME:       %[[IV0:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV1:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV2:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV3:[a-zA-Z0-9]+]],
//  CHECK-SAME:       %[[IV4:[a-zA-Z0-9]+]])
//  CHECK-SAME:       in (%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]])
//   CHECK-DAG:     "use1"(%[[IV0]], %[[IV1]])
//   CHECK-DAG:     "use2"(%[[IV2]], %[[IV3]], %[[IV4]])
//       CHECK:     mapping = [
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z:1>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z:2>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<y>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<x>
//  CHECK-SAME:         #iree_codegen.workgroup_mapping<z>]
