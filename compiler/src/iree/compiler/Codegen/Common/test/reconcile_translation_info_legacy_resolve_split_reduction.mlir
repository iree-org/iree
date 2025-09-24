// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info{fold-split-reduction-loop-into-workgroup-mapping-loop=false}, canonicalize, cse)))" %s --verify-diagnostics --allow-unregistered-dialect | FileCheck %s --enable-var-scope

// Test that the legacy approach to split-reduction loop resolution works.

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
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (((-s3 + s4) ceildiv s5) * (s2 * (s0 * s1)))>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 floordiv ((-s1 + s2) ceildiv s3))>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
//       CHECK: @split_reduction_variant
//       CHECK:   hal.executable.export
//  CHECK-SAME:       %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG5:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG6:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[NUMWORKGROUPSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG6]], %[[ARG5]], %[[ARG4]], %[[ARG1]], %[[ARG2]], %[[ARG3]]]
//       CHECK:     hal.return %[[NUMWORKGROUPSX]], %[[C1]], %[[C1]]
//       CHECK:   func @split_reduction()
//   CHECK-DAG:     %[[SPLIT_LB:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   CHECK-DAG:     %[[SPLIT_UB:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   CHECK-DAG:     %[[SPLIT_STEP:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//   CHECK-DAG:     %[[ORIG_UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   CHECK-DAG:     %[[ORIG_UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   CHECK-DAG:     %[[ORIG_UB2:.+]] = hal.interface.constant.load {{.+}} ordinal(5)
//   CHECK-DAG:     %[[SPLIT_NPROCS:.+]] = affine.apply #[[MAP1]]()[%[[SPLIT_LB]], %[[SPLIT_UB]], %[[SPLIT_STEP]]]
//   CHECK-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:     %[[COUNTX:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:     %[[ORIG_COUNTZ:.+]] = affine.apply #[[MAP2]]()[%[[COUNTX]], %[[SPLIT_LB]], %[[SPLIT_UB]], %[[SPLIT_STEP]]]
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IDX]] into (%[[SPLIT_NPROCS]], %[[ORIG_COUNTZ]]
//       CHECK:     %[[SPLITIVREPLACEMENT:.+]] = affine.apply #[[MAP3]]()[%[[DELINEARIZE]]#0, %[[SPLIT_STEP]], %[[SPLIT_LB]]]
//       CHECK:     "use1"(%[[SPLITIVREPLACEMENT]])
//       CHECK:     %[[OTHERIVREPLACEMENTS:.+]]:3 = affine.delinearize_index %[[DELINEARIZE]]#1
//  CHECK-SAME:       into (%[[ORIG_UB0]], %[[ORIG_UB1]], %[[ORIG_UB2]]
//       CHECK:     "use2"(%[[OTHERIVREPLACEMENTS]]#0, %[[OTHERIVREPLACEMENTS]]#1, %[[OTHERIVREPLACEMENTS]]#2)

// -----

// Test resolution of split reduction loop with rank > 1.

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly">,
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @split_reduction_2d_executable {
  hal.executable.variant public @split_reduction_2d_variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @split_reduction_2d layout(#pipeline_layout) count(
        %arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5)
      %return_x, %return_y, %return_z =
          iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%x, %y, %z) workload(%arg1, %arg2, %arg3, %arg4, %arg5)
      hal.return %return_x, %return_y, %return_z : index, index, index
    }
    builtin.module {
      func.func @split_reduction_2d() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cst4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %cst4, 4 : index
        scf.forall (%arg0, %arg1) in (%0, %1) {
          "use1"(%arg0, %arg1) : (index, index) -> ()
          scf.forall (%arg2, %arg3, %arg4) in (%2, %3, %4) {
            "use2"(%arg2, %arg3, %arg4) : (index, index, index) -> ()
          } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        } {mapping = [#iree_linalg_ext.split_reduction_mapping<1>, #iree_linalg_ext.split_reduction_mapping<0>]}
        return
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> ((s3 * s4) * (s2 * (s0 * s1)))>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 floordiv (s1 * s2))>
//       CHECK: @split_reduction_2d_variant
//       CHECK:   hal.executable.export
//  CHECK-SAME:       %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG5:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[NUMWORKGROUPSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG5]], %[[ARG4]], %[[ARG3]], %[[ARG1]], %[[ARG2]]]
//       CHECK:     hal.return %[[NUMWORKGROUPSX]], %[[C1]], %[[C1]]
//       CHECK:   func @split_reduction_2d()
//   CHECK-DAG:     %[[SPLIT_UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   CHECK-DAG:     %[[SPLIT_UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   CHECK-DAG:     %[[ORIG_UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//   CHECK-DAG:     %[[ORIG_UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   CHECK-DAG:     %[[ORIG_UB2:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   CHECK-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:     %[[COUNTX:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:     %[[ORIG_COUNTX:.+]] = affine.apply #[[MAP1]]()[%[[COUNTX]], %[[SPLIT_UB0]], %[[SPLIT_UB1]]]
//       CHECK:     %[[DELINEARIZE:.+]]:3 = affine.delinearize_index %[[IDX]] into (%[[SPLIT_UB0]], %[[SPLIT_UB1]], %[[ORIG_COUNTX]]
//       CHECK:     "use1"(%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1)
//       CHECK:     %[[OTHERIVREPLACEMENTS:.+]]:3 = affine.delinearize_index %[[DELINEARIZE]]#2
//  CHECK-SAME:       into (%[[ORIG_UB0]], %[[ORIG_UB1]], %[[ORIG_UB2]]
//       CHECK:     "use2"(%[[OTHERIVREPLACEMENTS]]#0, %[[OTHERIVREPLACEMENTS]]#1, %[[OTHERIVREPLACEMENTS]]#2)

// -----

// Test resolution of split reduction loop with permuted mappings.

#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly">,
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @split_reduction_2d_permuted_mapping_executable {
  hal.executable.variant public @split_reduction_2d_permuted_mapping_variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @split_reduction_2d_permuted_mapping layout(#pipeline_layout) count(
        %arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
      %return_x, %return_y, %return_z =
          iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%x, %y, %z) workload(%arg1, %arg2, %arg3, %arg4, %arg5, %arg6)
      hal.return %return_x, %return_y, %return_z : index, index, index
    }
    builtin.module {
      func.func @split_reduction_2d_permuted_mapping() {
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
        scf.forall (%arg0, %arg1, %arg2) in (%0, %1, %2) {
          "use1"(%arg0, %arg1, %arg2) : (index, index, index) -> ()
          scf.forall (%arg3, %arg4, %arg5) in (%3, %4, %5) {
            "use2"(%arg3, %arg4, %arg5) : (index, index, index) -> ()
          } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>]}
        } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>, #iree_linalg_ext.split_reduction_mapping<2>, #iree_linalg_ext.split_reduction_mapping<1>]}
        return
      }
    }
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (((s3 * s4) * s5) * (s2 * (s0 * s1)))>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 floordiv ((s1 * s2) * s3))>
//       CHECK: @split_reduction_2d_permuted_mapping_variant
//       CHECK:   hal.executable.export
//  CHECK-SAME:       %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG4:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG5:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:       %[[ARG6:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:     %[[NUMWORKGROUPSX:.+]] = affine.apply #[[MAP0]]()[%[[ARG6]], %[[ARG5]], %[[ARG4]], %[[ARG1]], %[[ARG2]], %[[ARG3]]]
//       CHECK:     hal.return %[[NUMWORKGROUPSX]], %[[C1]], %[[C1]]
//       CHECK:   func @split_reduction_2d_permuted_mapping()
//   CHECK-DAG:     %[[SPLIT_UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//   CHECK-DAG:     %[[SPLIT_UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//   CHECK-DAG:     %[[SPLIT_UB2:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//   CHECK-DAG:     %[[ORIG_UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//   CHECK-DAG:     %[[ORIG_UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(4)
//   CHECK-DAG:     %[[ORIG_UB2:.+]] = hal.interface.constant.load {{.+}} ordinal(5)
//   CHECK-DAG:     %[[IDX:.+]] = hal.interface.workgroup.id[0]
//   CHECK-DAG:     %[[COUNTX:.+]] = hal.interface.workgroup.count[0]
//   CHECK-DAG:     %[[ORIG_COUNTX:.+]] = affine.apply #[[MAP1]]()[%[[COUNTX]], %[[SPLIT_UB0]], %[[SPLIT_UB1]], %[[SPLIT_UB2]]]
//       CHECK:     %[[DELINEARIZE:.+]]:4 = affine.delinearize_index %[[IDX]] into (%[[SPLIT_UB0]], %[[SPLIT_UB1]], %[[SPLIT_UB2]], %[[ORIG_COUNTX]]
//       CHECK:     "use1"(%[[DELINEARIZE]]#2, %[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1)
//       CHECK:     %[[OTHERIVREPLACEMENTS:.+]]:3 = affine.delinearize_index %[[DELINEARIZE]]#3
//  CHECK-SAME:       into (%[[ORIG_UB1]], %[[ORIG_UB2]], %[[ORIG_UB0]]
//       CHECK:     "use2"(%[[OTHERIVREPLACEMENTS]]#2, %[[OTHERIVREPLACEMENTS]]#0, %[[OTHERIVREPLACEMENTS]]#1)
