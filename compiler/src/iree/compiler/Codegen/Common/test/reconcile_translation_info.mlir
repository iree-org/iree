// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info, canonicalize, cse)))" %s --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @reconcile_workgroup_size {
  hal.executable.variant public @reconcile_workgroup_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>}  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>} {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @reconcile_workgroup_size
//       CHECK: hal.executable.export public @entry_point
//  CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @single_translation_info {
  hal.executable.variant public @single_translation_info target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point()  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2()  {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @single_translation_info
//       CHECK: hal.executable.export public @entry_point
//  CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @err_mistmatched_workgroup_size {
  hal.executable.variant public @err_mismatched_workgroup_size target(#hal.executable.target<"", "", {}>) {
    // expected-error @+1 {{failed to reconcile workgroup sizes}}
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point()  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4, 2]>} {
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @err_mistmatched_workgroup_size2 {
  hal.executable.variant public @err_mismatched_workgroup_size2 target(#hal.executable.target<"", "", {}>) {
    // expected-error @+1 {{failed to reconcile workgroup sizes}}
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point()  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None>} {
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @reconcile_subgroup_size {
  hal.executable.variant public @reconcile_subgroup_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point()  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @reconcile_subgroup_size
//       CHECK: hal.executable.export public @entry_point
//  CHECK-SAME:     subgroup_size = 32 : index

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @err_reconcile_subgroup_size {
  hal.executable.variant public @err_reconcile_subgroup_size target(#hal.executable.target<"", "", {}>) {
    // expected-error @+1 {{failed to reconcile subgroup size}}
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point()  {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 64>} {
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @llvm_func_attrs {
  hal.executable.variant public @llvm_func_attrs target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout)
    hal.executable.export public @entry_point2 layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point1()  {
        func.call @fn1() : () -> ()
        return
      }
      func.func @entry_point2()  {
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None, {llvm_func_attrs = {"some-llvm-attr" = "2"}}>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None, {llvm_func_attrs = {"some-llvm-attr" = "4"}}>} {
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @llvm_func_attrs
//       CHECK:   func.func @fn1() attributes {llvm_func_attrs = {"some-llvm-attr" = "2"}}
//       CHECK:   func.func @fn2() attributes {llvm_func_attrs = {"some-llvm-attr" = "4"}}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_2D {
  hal.executable.variant public @scf_forall_2D target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_2D layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_2D() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        scf.forall (%arg0, %arg1) = (0, 0) to (%0, %1) step(64, 32) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 64)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 32)>
//      CHECK: hal.executable.export public @scf_forall_2D layout
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:   %[[WG_X:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[WG_X]], %[[WG_Y]], %[[C1]]
//      CHECK: func @scf_forall_2D()
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[I:.+]] = affine.apply #[[MAP2]]()[%[[WG_ID_Y]]]
//      CHECK:   %[[J:.+]] = affine.apply #[[MAP3]]()[%[[WG_ID_X]]]
//      CHECK:   "use"(%[[I]], %[[J]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_2D_dynamic_tile_size {
  hal.executable.variant public @scf_forall_2D_dynamic_tile_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_2D_dynamic_tile_size layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_2D_dynamic_tile_size() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        scf.forall (%arg0, %arg1) = (0, 0) to (%0, %1) step(%2, %3) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: hal.executable.export public @scf_forall_2D_dynamic_tile_size layout
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[WG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]], %[[ARG4]]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[WG_X]], %[[WG_Y]], %[[C1]]
//      CHECK: func @scf_forall_2D_dynamic_tile_size()
//  CHECK-DAG:   %[[STEP_Y:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//  CHECK-DAG:   %[[STEP_X:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[I:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_Y]], %[[STEP_Y]]]
//      CHECK:   %[[J:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]], %[[STEP_X]]]
//      CHECK:   "use"(%[[I]], %[[J]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 12, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_4D {
  hal.executable.variant public @scf_forall_4D target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_4D layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index,
                                                                               %arg5: index, %arg6: index, %arg7: index, %arg8: index,
                                                                               %arg9: index, %arg10: index, %arg11: index, %arg12: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3,
          %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_4D() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %cst4 = hal.interface.constant.load layout(#pipeline_layout) ordinal(4) : index
        %cst5 = hal.interface.constant.load layout(#pipeline_layout) ordinal(5) : index
        %cst6 = hal.interface.constant.load layout(#pipeline_layout) ordinal(6) : index
        %cst7 = hal.interface.constant.load layout(#pipeline_layout) ordinal(7) : index
        %cst8 = hal.interface.constant.load layout(#pipeline_layout) ordinal(8) : index
        %cst9 = hal.interface.constant.load layout(#pipeline_layout) ordinal(9) : index
        %cst10 = hal.interface.constant.load layout(#pipeline_layout) ordinal(10) : index
        %cst11 = hal.interface.constant.load layout(#pipeline_layout) ordinal(11) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        %4 = iree_tensor_ext.dispatch.workload.ordinal %cst4, 4 : index
        %5 = iree_tensor_ext.dispatch.workload.ordinal %cst5, 5 : index
        %6 = iree_tensor_ext.dispatch.workload.ordinal %cst6, 6 : index
        %7 = iree_tensor_ext.dispatch.workload.ordinal %cst7, 7 : index
        %8 = iree_tensor_ext.dispatch.workload.ordinal %cst8, 8 : index
        %9 = iree_tensor_ext.dispatch.workload.ordinal %cst9, 9 : index
        %10 = iree_tensor_ext.dispatch.workload.ordinal %cst10, 10 : index
        %11 = iree_tensor_ext.dispatch.workload.ordinal %cst11, 11 : index
        scf.forall (%arg0, %arg1, %arg2, %arg3) = (%0, %1, %2, %3) to (%4, %5, %6, %7) step(%8, %9, %10, %11) {
          "use"(%arg0, %arg1, %arg2, %arg3) : (index, index, index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<z:1>,
                      #iree_codegen.workgroup_mapping<z:0>,
                      #iree_codegen.workgroup_mapping<y>,
                      #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> ((-s0 + s1) ceildiv s2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (((-s0 + s1) ceildiv s2) * ((-s3 + s4) ceildiv s5))>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//      CHECK: hal.executable.export public @scf_forall_4D layout
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG6:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG7:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG8:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG9:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG10:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG11:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG12:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG3]], %[[ARG7]], %[[ARG11]]]
//  CHECK-DAG:   %[[WG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG4]], %[[ARG8]], %[[ARG12]]]
//  CHECK-DAG:   %[[WG_Z:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[ARG6]], %[[ARG10]], %[[ARG1]], %[[ARG5]], %[[ARG9]]]
//      CHECK:   hal.return %[[WG_X]], %[[WG_Y]], %[[WG_Z]]
//      CHECK: func @scf_forall_4D()
//  CHECK-DAG:   %[[LB0:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(0)
//  CHECK-DAG:   %[[LB1:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(1)
//  CHECK-DAG:   %[[UB0:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(4)
//  CHECK-DAG:   %[[UB1:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(5)
//  CHECK-DAG:   %[[STEP0:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(8)
//  CHECK-DAG:   %[[STEP1:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(9)
//  CHECK-DAG:   %[[STEP2:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(10)
//  CHECK-DAG:   %[[STEP3:.+]] = hal.interface.constant.load layout(#{{.+}}) ordinal(11)
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[NITERS1:.+]] = affine.apply #[[MAP0]]()[%[[LB1]], %[[UB1]], %[[STEP1]]]
//  CHECK-DAG:   %[[NITERS0:.+]] = affine.apply #[[MAP0]]()[%[[LB0]], %[[UB0]], %[[STEP0]]]
//  CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[WG_ID_Z]] into (%[[NITERS0]], %[[NITERS1]])
//      CHECK:   %[[I:.+]] = affine.apply #[[MAP2]]()[%[[DELINEARIZE]]#0, %[[STEP0]]]
//      CHECK:   %[[J:.+]] = affine.apply #[[MAP2]]()[%[[DELINEARIZE]]#1, %[[STEP1]]]
//      CHECK:   %[[K:.+]] = affine.apply #[[MAP2]]()[%[[WG_ID_Y]], %[[STEP2]]]
//      CHECK:   %[[L:.+]] = affine.apply #[[MAP2]]()[%[[WG_ID_X]], %[[STEP3]]]
//      CHECK:   "use"(%[[I]], %[[J]], %[[K]], %[[L]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_4D_static_interchange {
  hal.executable.variant public @scf_forall_4D_static_interchange target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_4D_static_interchange layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_4D_static_interchange() {
        scf.forall (%arg0, %arg1, %arg2, %arg3, %arg4) = (0, 1, 2, 3, 4) to (4, 10, 19, 29, 44) step(1, 2, 3, 4, 5) {
          "use"(%arg0, %arg1, %arg2, %arg3, %arg4) : (index, index, index, index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<z:0>,
                      #iree_codegen.workgroup_mapping<z:2>,
                      #iree_codegen.workgroup_mapping<x>,
                      #iree_codegen.workgroup_mapping<y>,
                      #iree_codegen.workgroup_mapping<z:1>]}
        return
      }
    }
  }
}
//  CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
//  CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 * 3)>
//  CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0] -> (s0 * 4)>
//  CHECK-DAG:  #[[MAP3:.+]] = affine_map<()[s0] -> (s0 * 5)>
//      CHECK: hal.executable.export public @scf_forall_4D_static_interchange layout
//  CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C160:.+]] = arith.constant 160 : index
//      CHECK:   hal.return %[[C6]], %[[C7]], %[[C160]]
//      CHECK: func @scf_forall_4D_static_interchange()
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[DELINEARIZE:.+]]:3 = affine.delinearize_index %[[WG_ID_Z]] into (5, 8, 4)
//      CHECK:   %[[I:.+]] = affine.apply #[[MAP0]]()[%[[DELINEARIZE]]#0]
//      CHECK:   %[[J:.+]] = affine.apply #[[MAP1]]()[%[[WG_ID_X]]]
//      CHECK:   %[[K:.+]] = affine.apply #[[MAP2]]()[%[[WG_ID_Y]]]
//      CHECK:   %[[L:.+]] = affine.apply #[[MAP3]]()[%[[DELINEARIZE]]#1]
//      CHECK:   "use"(%[[DELINEARIZE]]#2, %[[I]], %[[J]], %[[K]], %[[L]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @no_loop_do_nothing {
  hal.executable.variant public @no_loop_do_nothing target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @no_loop_do_nothing layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1: index
      %c2 = arith.constant 2: index
      hal.return %c1, %c2, %c1 : index, index, index
    }
    builtin.module {
      func.func @no_loop_do_nothing() {
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @no_loop_do_nothing layout
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   hal.return %[[C1]], %[[C2]], %[[C1]]
//      CHECK: func @no_loop_do_nothing()
// CHECK-NEXT:   return

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @no_loop_default_workgroup_count {
  hal.executable.variant public @no_loop_default_workgroup_count target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @no_loop_default_workgroup_count layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index) -> (index, index, index) {
      %0:3 = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2
      hal.return %0#1, %0#2, %0#0 : index, index, index
    }
    builtin.module {
      func.func @no_loop_default_workgroup_count() {
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @no_loop_default_workgroup_count layout
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]]
//      CHECK: func @no_loop_default_workgroup_count()
// CHECK-NEXT:   return

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @multi_export_scf_forall {
  hal.executable.variant public @multi_export_scf_forall target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point0 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @entry_point1 layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point0() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        scf.forall (%arg0, %arg1) = (0, 0) to (%0, %1) step(64, 32) {
          "use0"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
      func.func @entry_point1() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        scf.forall (%arg0, %arg1) = (0, 0) to (%0, %1) step(64, 32) {
          "use1"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)
//      CHECK: hal.executable.export public @entry_point0 layout
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]]]
//  CHECK-DAG:   %[[WG_X:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//      CHECK:   hal.return %[[WG_X]], %[[WG_Y]], %[[C1]]
//      CHECK: hal.executable.export public @entry_point1 layout
// CHECK-SAME:     %[[ARG1_1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2_1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3_1:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y_1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1_1]]]
//  CHECK-DAG:   %[[WG_X_1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2_1]]]
//  CHECK-DAG:   %[[C1_1:.+]] = arith.constant 1
//      CHECK:   hal.return %[[WG_X_1]], %[[WG_Y_1]], %[[C1_1]]

//      CHECK: func @entry_point0()
//  CHECK-DAG:   hal.interface.workgroup.id[1]
//  CHECK-DAG:   hal.interface.workgroup.id[0]
//  CHECK-NOT:   scf.forall
//      CHECK:   "use0"
//      CHECK: func @entry_point1()
//  CHECK-DAG:   hal.interface.workgroup.id[1]
//  CHECK-DAG:   hal.interface.workgroup.id[0]
//  CHECK-NOT:   scf.forall
//      CHECK:   "use1"

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @multiple_scf_forall_ops {
  hal.executable.variant public @multiple_scf_forall_ops target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @multiple_scf_forall_ops layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @multiple_scf_forall_ops() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %cst2, 2 : index
        %3 = iree_tensor_ext.dispatch.workload.ordinal %cst3, 3 : index
        scf.forall (%arg0, %arg1) = (0, 0) to (%0, %1) step(%2, %3) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        scf.forall (%arg0, %arg1) = (0, 0) to (16, 32) step(4, 2) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2, s3] -> ((s2 ceildiv s3) * (s0 ceildiv s1))>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s0 ceildiv s1, s2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 ceildiv s1, s2 ceildiv s3)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<()[s0] -> (4, s0)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<()[s0, s1] -> (16, s0 ceildiv s1)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<()[s0] -> (s0 * 4)>
//  CHECK-DAG: #[[MAP8:.+]] = affine_map<()[s0] -> (s0 * 2)>
//      CHECK: hal.executable.export public @multiple_scf_forall_ops layout
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_X0:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]], %[[ARG2]], %[[ARG4]]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[WG_X1:.+]] = arith.constant 64
//  CHECK-DAG:   %[[WG_X:.+]] = arith.maxui %[[WG_X0]], %[[WG_X1]]
//      CHECK:   hal.return %[[WG_X]], %[[C1]], %[[C1]]
//      CHECK: func @multiple_scf_forall_ops()
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32
//  CHECK-DAG:   %[[C16:.+]] = arith.constant 16
//  CHECK-DAG:   %[[UB0:.+]] = hal.interface.constant.load {{.+}} ordinal(0)
//  CHECK-DAG:   %[[UB1:.+]] = hal.interface.constant.load {{.+}} ordinal(1)
//  CHECK-DAG:   %[[STEP0:.+]] = hal.interface.constant.load {{.+}} ordinal(2)
//  CHECK-DAG:   %[[STEP1:.+]] = hal.interface.constant.load {{.+}} ordinal(3)
//  CHECK-DAG:   %[[EXTENT0:.+]] = affine.apply #[[MAP1]]()[%[[UB0]], %[[STEP0]]]
//  CHECK-DAG:   %[[EXTENT1:.+]] = affine.apply #[[MAP1]]()[%[[UB1]], %[[STEP1]]]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]

//  CHECK-DAG:   %[[WG_IDS_1ST:.+]]:2 = affine.delinearize_index %[[WG_ID_X]]
// CHECK-SAME:     into (%[[EXTENT0]], %[[EXTENT1]])
//  CHECK-DAG:   %[[NPROCS1_1ST:.+]] = affine.min #[[MAP2]]()[%[[UB1]], %[[STEP1]], %[[WG_COUNT_X]]]
//  CHECK-DAG:   %[[NPROCS0_1ST:.+]] = affine.min #[[MAP3]]()[%[[UB0]], %[[STEP0]], %[[WG_COUNT_X]], %[[NPROCS1_1ST]]]
//  CHECK-DAG:   %[[LOOP_START0_1ST:.+]] = affine.apply #[[MAP4]]()[%[[WG_IDS_1ST]]#0, %[[STEP0]]]
//  CHECK-DAG:   %[[LOOP_START1_1ST:.+]] = affine.apply #[[MAP4]]()[%[[WG_IDS_1ST]]#1, %[[STEP1]]]
//  CHECK-DAG:   %[[LOOP_STEP0_1ST:.+]] = affine.apply #[[MAP4]]()[%[[NPROCS0_1ST]], %[[STEP0]]]
//  CHECK-DAG:   %[[LOOP_STEP1_1ST:.+]] = affine.apply #[[MAP4]]()[%[[NPROCS1_1ST]], %[[STEP1]]]
//  CHECK-NOT:   scf.forall
//      CHECK:   scf.for %[[I:.+]] = %[[LOOP_START0_1ST]] to %[[UB0]] step %[[LOOP_STEP0_1ST]]
//      CHECK:     scf.for %[[J:.+]] = %[[LOOP_START1_1ST]] to %[[UB1]] step %[[LOOP_STEP1_1ST]]
//      CHECK:       "use"(%[[I]], %[[J]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }

//  CHECK-DAG:   %[[WG_IDS_2ND:.+]]:2 = affine.delinearize_index %[[WG_ID_X]] into (16, 4)
//  CHECK-DAG:   %[[NPROCS1_2ND:.+]] = affine.min #[[MAP5]]()[%[[WG_COUNT_X]]]
//  CHECK-DAG:   %[[NPROCS0_2ND:.+]] = affine.min #[[MAP6]]()[%[[WG_COUNT_X]], %[[NPROCS1_2ND]]]
//  CHECK-DAG:   %[[LOOP_START1_2ND:.+]] = affine.apply #[[MAP8]]()[%[[WG_IDS_2ND]]#0]
//  CHECK-DAG:   %[[LOOP_START0_2ND:.+]] = affine.apply #[[MAP7]]()[%[[WG_IDS_2ND]]#1]
//  CHECK-DAG:   %[[LOOP_STEP1_2ND:.+]] = affine.apply #[[MAP8]]()[%[[NPROCS0_2ND]]]
//  CHECK-DAG:   %[[LOOP_STEP0_2ND:.+]] = affine.apply #[[MAP7]]()[%[[NPROCS1_2ND]]]
//      CHECK:   scf.for %[[I:.+]] = %[[LOOP_START0_2ND]] to %[[C16]] step %[[LOOP_STEP0_2ND]]
//      CHECK:     scf.for %[[J:.+]] = %[[LOOP_START1_2ND]] to %[[C32]] step %[[LOOP_STEP1_2ND]]
//      CHECK:       "use"(%[[I]], %[[J]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @different_rank_scf_forall_ops {
  hal.executable.variant public @different_rank_scf_forall_ops target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @different_rank_scf_forall_ops layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @different_rank_scf_forall_ops() {
        scf.forall (%arg0, %arg1) = (0, 0) to (2, 42) step(1, 1) {
          "use"(%arg0, %arg1) : (index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        scf.forall (%arg0, %arg1, %arg2) = (0, 0, 0) to (4, 2, 8) step(1, 1, 1) {
          "use"(%arg0, %arg1, %arg2) : (index, index, index) -> ()
          scf.forall.in_parallel {}
        } {mapping = [#iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
        return
      }
    }
  }
}
//      CHECK: hal.executable.export public @different_rank_scf_forall_ops
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[WG_X:.+]] = arith.constant 84
//      CHECK:   hal.return %[[WG_X]], %[[C1]], %[[C1]]
//      CHECK: func @different_rank_scf_forall_ops
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4
//  CHECK-DAG:   %[[C8:.+]] = arith.constant 8
//  CHECK-DAG:   %[[C42:.+]] = arith.constant 42
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_COUNT_X:.+]] = hal.interface.workgroup.count[0]

//  CHECK-DAG:   %[[WG_IDS_1ST:.+]]:2 = affine.delinearize_index %[[WG_ID_X]] into (2, 42)
//  CHECK-NOT:   scf.forall
//      CHECK:   scf.for %[[I:.+]] = %[[WG_IDS_1ST]]#0 to %[[C2]]
//      CHECK:     scf.for %[[J:.+]] = %[[WG_IDS_1ST]]#1 to %[[C42]]
//      CHECK:       "use"(%[[I]], %[[J]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }

//  CHECK-DAG:   %[[WG_IDS_2ND:.+]]:3 = affine.delinearize_index %[[WG_ID_X]] into (4, 2, 8)
//      CHECK:   scf.for %[[I:.+]] = %[[WG_IDS_2ND]]#0 to %[[C4]]
//      CHECK:     scf.for %[[J:.+]] = %[[WG_IDS_2ND]]#1 to %[[C2]]
//      CHECK:       scf.for %[[K:.+]] = %[[WG_IDS_2ND]]#2 to %[[C8]]
//      CHECK:         "use"(%[[I]], %[[J]], %[[K]])
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
