// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info{distribute-along=x fold-split-reduction-loop-into-workgroup-mapping-loop=false}, canonicalize)))" --allow-unregistered-dialect --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,DISTRIBUTEX --enable-var-scope
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info{distribute-along=y fold-split-reduction-loop-into-workgroup-mapping-loop=false}, canonicalize)))" --allow-unregistered-dialect --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,DISTRIBUTEY --enable-var-scope
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info{distribute-along=z fold-split-reduction-loop-into-workgroup-mapping-loop=false}, canonicalize)))" --allow-unregistered-dialect --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK-ALL,DISTRIBUTEZ --enable-var-scope

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_2D_dynamic_tile_size {
  hal.executable.variant public @scf_forall_2D_dynamic_tile_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_2D_dynamic_tile_size layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_2D_dynamic_tile_size() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cst4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %cst5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %cst4, 4 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %cst5, 5 : index
        scf.forall (%arg0, %arg1) = (%0, %1) to (%2, %3) step(%4, %5) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//         CHECK-ALL:   hal.executable.export
//    CHECK-ALL-SAME:       %[[ARG0:[a-zA-Z0-9]+]]: !hal.device
//    CHECK-ALL-SAME:       %[[ARG1:[a-zA-Z0-9]+]]: index
//    CHECK-ALL-SAME:       %[[ARG2:[a-zA-Z0-9]+]]: index
//    CHECK-ALL-SAME:       %[[ARG3:[a-zA-Z0-9]+]]: index
//    CHECK-ALL-SAME:       %[[ARG4:[a-zA-Z0-9]+]]: index
//    CHECK-ALL-SAME:       %[[ARG5:[a-zA-Z0-9]+]]: index
//    CHECK-ALL-SAME:       %[[ARG6:[a-zA-Z0-9]+]]: index

//       DISTRIBUTEX:     %[[C1:.+]] = arith.constant 1 : index
//       DISTRIBUTEX:     %[[SIZE:.+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> (((-s3 + s4) ceildiv s5) * ((-s0 + s1) ceildiv s2))>()
//  DISTRIBUTEX-SAME:         [%[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG1]], %[[ARG3]], %[[ARG5]]]
//       DISTRIBUTEX:     hal.return %[[SIZE]], %[[C1]], %[[C1]]

//       DISTRIBUTEY:     %[[C1:.+]] = arith.constant 1 : index
//   DISTRIBUTEY-DAG:     %[[SIZE0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG1]], %[[ARG3]], %[[ARG5]]]
//   DISTRIBUTEY-DAG:     %[[SIZE1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG2]], %[[ARG4]], %[[ARG6]]]
//       DISTRIBUTEY:     hal.return %[[SIZE1]], %[[SIZE0]], %[[C1]]

//       DISTRIBUTEZ:     %[[C1:.+]] = arith.constant 1 : index
//   DISTRIBUTEZ-DAG:     %[[SIZE0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG1]], %[[ARG3]], %[[ARG5]]]
//   DISTRIBUTEZ-DAG:     %[[SIZE1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG2]], %[[ARG4]], %[[ARG6]]]
//       DISTRIBUTEZ:     hal.return %[[SIZE1]], %[[SIZE0]], %[[C1]]

//         CHECK-ALL:   @scf_forall_2D_dynamic_tile_size()

//   DISTRIBUTEX-DAG:     %[[ARG0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   DISTRIBUTEX-DAG:     %[[ARG1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   DISTRIBUTEX-DAG:     %[[ARG2:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//   DISTRIBUTEX-DAG:     %[[ARG3:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   DISTRIBUTEX-DAG:     %[[ARG4:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   DISTRIBUTEX-DAG:     %[[ARG5:.+]] = hal.interface.constant.load {{.+}} ordinal(5)
//   DISTRIBUTEX-DAG:     %[[SIZE0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG1]], %[[ARG3]], %[[ARG5]]]
//   DISTRIBUTEX-DAG:     %[[SIZE1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[ARG0]], %[[ARG2]], %[[ARG4]]]
//   DISTRIBUTEX-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEX-DAG:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX]] into (%[[SIZE1]], %[[SIZE0]])
//   DISTRIBUTEX-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s2 + s1)>()[%[[ARG4]], %[[ARG0]], %[[DELINEARIZE]]#0]
//   DISTRIBUTEX-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s2 + s1)>()[%[[ARG5]], %[[ARG1]], %[[DELINEARIZE]]#1]
//       DISTRIBUTEX:     "use"(%[[IV0]], %[[IV1]])

//   DISTRIBUTEY-DAG:     %[[ARG0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   DISTRIBUTEY-DAG:     %[[ARG1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   DISTRIBUTEY-DAG:     %[[ARG4:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   DISTRIBUTEY-DAG:     %[[ARG5:.+]] = hal.interface.constant.load {{.+}} ordinal(5)
//   DISTRIBUTEY-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEY-DAG:     %[[IDY:.+]] = hal.interface.workgroup.id[1] : index
//   DISTRIBUTEY-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[IDY]], %[[ARG4]], %[[ARG0]]]
//   DISTRIBUTEY-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[IDX]], %[[ARG5]], %[[ARG1]]]
//       DISTRIBUTEY:     "use"(%[[IV0]], %[[IV1]])

//   DISTRIBUTEZ-DAG:     %[[ARG0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   DISTRIBUTEZ-DAG:     %[[ARG1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   DISTRIBUTEZ-DAG:     %[[ARG4:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   DISTRIBUTEZ-DAG:     %[[ARG5:.+]] = hal.interface.constant.load {{.+}} ordinal(5)
//   DISTRIBUTEZ-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEZ-DAG:     %[[IDY:.+]] = hal.interface.workgroup.id[1] : index
//   DISTRIBUTEZ-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[IDY]], %[[ARG4]], %[[ARG0]]]
//   DISTRIBUTEZ-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[IDX]], %[[ARG5]], %[[ARG1]]]
//       DISTRIBUTEZ:     "use"(%[[IV0]], %[[IV1]])

// // -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_3D_tile_size {
  hal.executable.variant public @scf_forall_3D_tile_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_3D_tile_size layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_3D_tile_size() {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c44 = arith.constant 44 : index
        %c55 = arith.constant 55 : index
        %c66 = arith.constant 66 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        scf.forall (%arg0, %arg1, %arg2) = (%c1, %c2, %c3) to (%c44, %c55, %c66) step(%c7, %c6, %c5) {
          "use"(%arg0, %arg1, %arg2) : (index, index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}

//         CHECK-ALL:   hal.executable.export

//   DISTRIBUTEX-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   DISTRIBUTEX-DAG:     %[[C819:.+]] = arith.constant 819 : index
//       DISTRIBUTEX:     hal.return %[[C819]], %[[C1]], %[[C1]]

//   DISTRIBUTEY-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   DISTRIBUTEY-DAG:     %[[C13:.+]] = arith.constant 13 : index
//   DISTRIBUTEY-DAG:     %[[C63:.+]] = arith.constant 63 : index
//       DISTRIBUTEY:     hal.return %[[C13]], %[[C63]], %[[C1]]

//   DISTRIBUTEZ-DAG:     %[[C13:.+]] = arith.constant 13 : index
//   DISTRIBUTEZ-DAG:     %[[C9:.+]] = arith.constant 9 : index
//   DISTRIBUTEZ-DAG:     %[[C7:.+]] = arith.constant 7 : index
//       DISTRIBUTEZ:     hal.return %[[C13]], %[[C9]], %[[C7]]

//         CHECK-ALL:   @scf_forall_3D_tile_size()

//   DISTRIBUTEX-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEX-DAG:     %[[DELINEARIZE:.+]]:3 = affine.delinearize_index %[[IDX]] into (7, 9, 13)
//   DISTRIBUTEX-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0] -> (s0 * 7 + 1)>()[%[[DELINEARIZE]]#0]
//   DISTRIBUTEX-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0] -> (s0 * 6 + 2)>()[%[[DELINEARIZE]]#1]
//   DISTRIBUTEX-DAG:     %[[IV2:.+]] = affine.apply affine_map<()[s0] -> (s0 * 5 + 3)>()[%[[DELINEARIZE]]#2]
//       DISTRIBUTEX:     "use"(%[[IV0]], %[[IV1]], %[[IV2]])

//   DISTRIBUTEY-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEY-DAG:     %[[IDY:.+]] = hal.interface.workgroup.id[1] : index
//   DISTRIBUTEY-DAG:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDY]] into (7, 9)
//   DISTRIBUTEY-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0] -> (s0 * 7 + 1)>()[%[[DELINEARIZE]]#0]
//   DISTRIBUTEY-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0] -> (s0 * 6 + 2)>()[%[[DELINEARIZE]]#1]
//   DISTRIBUTEY-DAG:     %[[IV2:.+]] = affine.apply affine_map<()[s0] -> (s0 * 5 + 3)>()[%[[IDX]]]
//       DISTRIBUTEY:     "use"(%[[IV0]], %[[IV1]], %[[IV2]])

//   DISTRIBUTEZ-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0] : index
//   DISTRIBUTEZ-DAG:     %[[IDY:.+]] = hal.interface.workgroup.id[1] : index
//   DISTRIBUTEZ-DAG:     %[[IDZ:.+]] = hal.interface.workgroup.id[2] : index
//   DISTRIBUTEZ-DAG:     %[[IV0:.+]] = affine.apply affine_map<()[s0] -> (s0 * 7 + 1)>()[%[[IDZ]]]
//   DISTRIBUTEZ-DAG:     %[[IV1:.+]] = affine.apply affine_map<()[s0] -> (s0 * 6 + 2)>()[%[[IDY]]]
//   DISTRIBUTEZ-DAG:     %[[IV2:.+]] = affine.apply affine_map<()[s0] -> (s0 * 5 + 3)>()[%[[IDX]]]
//       DISTRIBUTEZ:     "use"(%[[IV0]], %[[IV1]], %[[IV2]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly">,
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @split_reduction_executable {
  hal.executable.variant public @split_reduction_variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @split_reduction layout(#pipeline_layout) count(
        %arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
      %return_x, %return_y, %return_z =
          iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%x, %y, %z) workload(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
      hal.return %return_x, %return_y, %return_z : index, index, index
    }
    builtin.module {
      func.func @split_reduction() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cst4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %cst5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %cst4, 4 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %cst5, 5 : index
        scf.forall (%arg0) = (%0) to (%1) step (%2) {
          "use1"(%arg0) : (index) -> ()
          scf.forall (%arg1, %arg2, %arg3) in (%3, %4, %5) {
            "use2"(%arg1, %arg2, %arg3) : (index, index, index) -> ()
          } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
        return
      }
    }
  }
}
//         CHECK-ALL: @split_reduction_variant
//         CHECK-ALL:   hal.executable.export
//    CHECK-ALL-SAME:       %[[ARG1:[a-zA-Z0-9_]+]]: index
//    CHECK-ALL-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: index
//    CHECK-ALL-SAME:       %[[ARG3:[a-zA-Z0-9_]+]]: index
//    CHECK-ALL-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: index
//    CHECK-ALL-SAME:       %[[ARG5:[a-zA-Z0-9_]+]]: index
//    CHECK-ALL-SAME:       %[[ARG6:[a-zA-Z0-9_]+]]: index

//   DISTRIBUTEX-DAG:     %[[C1:.+]] = arith.constant 1 : index
//       DISTRIBUTEX:     %[[NUMWORKGROUPSX:.+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5] -> (((-s3 + s4) ceildiv s5) * (s2 * (s0 * s1)))>
//  DISTRIBUTEX-SAME:         [%[[ARG6]], %[[ARG5]], %[[ARG4]], %[[ARG1]], %[[ARG2]], %[[ARG3]]]
//       DISTRIBUTEX:     hal.return %[[NUMWORKGROUPSX]], %[[C1]], %[[C1]]

//   DISTRIBUTEY-DAG:     %[[C1:.+]] = arith.constant 1 : index
//       DISTRIBUTEY:     %[[NUMWORKGROUPSY:.+]] = affine.apply affine_map<()[s0, s1, s2, s3, s4] -> (((-s2 + s3) ceildiv s4) * (s0 * s1))>
//  DISTRIBUTEY-SAME:         [%[[ARG5]], %[[ARG4]], %[[ARG1]], %[[ARG2]], %[[ARG3]]]
//       DISTRIBUTEY:     hal.return %[[ARG6]], %[[NUMWORKGROUPSY]], %[[C1]]

//       DISTRIBUTEZ:     %[[NUMWORKGROUPSZ:.+]] = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 * ((-s0 + s1) ceildiv s2))>
//  DISTRIBUTEZ-SAME:         (%[[ARG4]])[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
//       DISTRIBUTEZ:     hal.return %[[ARG6]], %[[ARG5]], %[[NUMWORKGROUPSZ]]

//         CHECK-ALL:   func @split_reduction
//     CHECK-ALL-DAG:     %[[SPLIT_LB:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//     CHECK-ALL-DAG:     %[[SPLIT_UB:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//     CHECK-ALL-DAG:     %[[SPLIT_STEP:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//   DISTRIBUTEX-DAG:     %[[WG_BOUND_Z:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   DISTRIBUTEX-DAG:     %[[WG_BOUND_Y:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   DISTRIBUTEX-DAG:     %[[WG_BOUND_X:.+]] = hal.interface.constant.load {{.+}} ordinal(5)

//   DISTRIBUTEY-DAG:     %[[WG_BOUND_Z:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   DISTRIBUTEY-DAG:     %[[WG_BOUND_Y:.+]] = hal.interface.constant.load {{.+}} ordinal(4)

//     CHECK-ALL-DAG:     %[[SPLIT_NPROCS:.+]] = affine.apply affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>()[%[[SPLIT_LB]], %[[SPLIT_UB]], %[[SPLIT_STEP]]]

//   DISTRIBUTEX-DAG:     %[[DELINEARIZE_FROM_ID:.+]] = hal.interface.workgroup.id[0]
//   DISTRIBUTEX-DAG:     %[[DELINEARIZE_FROM_COUNT:.+]] = hal.interface.workgroup.count[0]

//   DISTRIBUTEY-DAG:     %[[DELINEARIZE_FROM_ID:.+]] = hal.interface.workgroup.id[1]
//   DISTRIBUTEY-DAG:     %[[DELINEARIZE_FROM_COUNT:.+]] = hal.interface.workgroup.count[1]

//   DISTRIBUTEZ-DAG:     %[[DELINEARIZE_FROM_ID:.+]] = hal.interface.workgroup.id[2]
//   DISTRIBUTEZ-DAG:     %[[DELINEARIZE_FROM_COUNT:.+]] = hal.interface.workgroup.count[2]

//         CHECK-ALL:     %[[ORIG_NPROCS:.+]] = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 floordiv ((-s1 + s2) ceildiv s3))>
//    CHECK-ALL-SAME:         ()[%[[DELINEARIZE_FROM_COUNT]], %[[SPLIT_LB]], %[[SPLIT_UB]], %[[SPLIT_STEP]]]
//         CHECK-ALL:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[DELINEARIZE_FROM_ID]] into (%[[SPLIT_NPROCS]], %[[ORIG_NPROCS]])
//         CHECK-ALL:     %[[SPLITIVREPLACEMENT:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[DELINEARIZE]]#0, %[[SPLIT_STEP]], %[[SPLIT_LB]]]
//         CHECK-ALL:     "use1"(%[[SPLITIVREPLACEMENT]])

//       DISTRIBUTEX:     %[[DELINEARIZE2:.+]]:3 = affine.delinearize_index %[[DELINEARIZE]]#1 into (%[[WG_BOUND_Z]], %[[WG_BOUND_Y]], %[[WG_BOUND_X]])
//       DISTRIBUTEX:     "use2"(%[[DELINEARIZE2]]#0, %[[DELINEARIZE2]]#1, %[[DELINEARIZE2]]#2)

//   DISTRIBUTEY-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0]
//   DISTRIBUTEY-DAG:     %[[DELINEARIZE2:.+]]:2 = affine.delinearize_index %[[DELINEARIZE]]#1 into (%[[WG_BOUND_Z]], %[[WG_BOUND_Y]])
//       DISTRIBUTEY:     "use2"(%[[DELINEARIZE2]]#0, %[[DELINEARIZE2]]#1, %[[IDX]])

//   DISTRIBUTEZ-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0]
//   DISTRIBUTEZ-DAG:     %[[IDY:.+]] = hal.interface.workgroup.id[1]
//       DISTRIBUTEZ:     "use2"(%[[DELINEARIZE]]#1, %[[IDY]], %[[IDX]])
