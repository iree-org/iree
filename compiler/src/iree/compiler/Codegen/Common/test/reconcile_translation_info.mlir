// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info, canonicalize)))" %s --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @err_multiple_entry_point {
  // expected-error @+1 {{reconciliation for multiple export ops unsupported}}
  hal.executable.variant public @reconcile_workgroup_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout)
    hal.executable.export public @entry_point2 layout(#pipeline_layout)
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @reconcile_workgroup_size {
  hal.executable.variant public @reconcile_workgroup_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4]>} {
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
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4]>}  {
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
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4, 2]>} {
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
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [4]>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None>} {
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
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>} {
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
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>} {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @err_reconcile_subgroup_size
//       CHECK: hal.executable.export public @entry_point
//  CHECK-SAME:     subgroup_size = 32 : index

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @llvm_func_attrs {
  hal.executable.variant public @llvm_func_attrs target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}>} {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @llvm_func_attrs
//       CHECK:   func.func @fn1() attributes {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}
//       CHECK:   func.func @fn2() attributes {llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_2D {
  hal.executable.variant public @scf_forall_2D target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_2D layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_2D() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %0 = flow.dispatch.workload.ordinal %cst0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cst1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cst2, 2 : index
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
//      CHECK: hal.executable.export public @scf_forall_2D layout
// CHECK-NEXT:     %[[ARG1:[a-zA-z0-9]+]]: index
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
//      CHECK:   "use"(%[[WG_ID_Y]], %[[WG_ID_X]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_2D_dynamic_tile_size {
  hal.executable.variant public @scf_forall_2D_dynamic_tile_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_2D_dynamic_tile_size layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @scf_forall_2D_dynamic_tile_size() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %cst2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %cst3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = flow.dispatch.workload.ordinal %cst0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cst1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cst2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cst3, 3 : index
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
//      CHECK: hal.executable.export public @scf_forall_2D_dynamic_tile_size layout
// CHECK-NEXT:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-z0-9]+]]: index
//  CHECK-DAG:   %[[WG_Y:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[ARG3]]]
//  CHECK-DAG:   %[[WG_X:.+]] = affine.apply #[[MAP0]]()[%[[ARG2]], %[[ARG4]]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   hal.return %[[WG_X]], %[[WG_Y]], %[[C1]]
//      CHECK: func @scf_forall_2D_dynamic_tile_size()
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-NOT:   scf.forall
//      CHECK:   "use"(%[[WG_ID_Y]], %[[WG_ID_X]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 12, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_4D {
  hal.executable.variant public @scf_forall_4D target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_4D layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index, %arg4 : index,
         %arg5 : index, %arg6 : index, %arg7 : index, %arg8 : index,
         %arg9 : index, %arg10 : index, %arg11 : index, %arg12 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3,
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
        %0 = flow.dispatch.workload.ordinal %cst0, 0 : index
        %1 = flow.dispatch.workload.ordinal %cst1, 1 : index
        %2 = flow.dispatch.workload.ordinal %cst2, 2 : index
        %3 = flow.dispatch.workload.ordinal %cst3, 3 : index
        %4 = flow.dispatch.workload.ordinal %cst4, 4 : index
        %5 = flow.dispatch.workload.ordinal %cst5, 5 : index
        %6 = flow.dispatch.workload.ordinal %cst6, 6 : index
        %7 = flow.dispatch.workload.ordinal %cst7, 7 : index
        %8 = flow.dispatch.workload.ordinal %cst8, 8 : index
        %9 = flow.dispatch.workload.ordinal %cst9, 9 : index
        %10 = flow.dispatch.workload.ordinal %cst10, 10 : index
        %11 = flow.dispatch.workload.ordinal %cst11, 11 : index
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
//      CHECK: hal.executable.export public @scf_forall_4D layout
// CHECK-NEXT:     %[[ARG1:[a-zA-z0-9]+]]: index
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
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[NITERS1:.+]] = affine.apply #[[MAP0]]()[%[[LB1]], %[[UB1]], %[[STEP1]]]
//  CHECK-DAG:   %[[NITERS0:.+]] = affine.apply #[[MAP0]]()[%[[LB0]], %[[UB0]], %[[STEP0]]]
//  CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[WG_ID_Z]] into (%[[NITERS0]], %[[NITERS1]])
//      CHECK:   "use"(%[[DELINEARIZE]]#0, %[[DELINEARIZE]]#1, %[[WG_ID_Y]], %[[WG_ID_X]])

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @scf_forall_4D_static_interchange {
  hal.executable.variant public @scf_forall_4D_static_interchange target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @scf_forall_4D_static_interchange layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
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
//      CHECK: hal.executable.export public @scf_forall_4D_static_interchange layout
//  CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//  CHECK-DAG:   %[[C7:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[C160:.+]] = arith.constant 160 : index
//      CHECK:   hal.return %[[C6]], %[[C7]], %[[C160]]
//      CHECK: func @scf_forall_4D_static_interchange()
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//  CHECK-DAG:   %[[WG_ID_X:.+]] = hal.interface.workgroup.id[0]
//  CHECK-DAG:   %[[WG_ID_Y:.+]] = hal.interface.workgroup.id[1]
//  CHECK-DAG:   %[[WG_ID_Z:.+]] = hal.interface.workgroup.id[2]
//  CHECK-NOT:   scf.forall
//      CHECK:   %[[DELINEARIZE:.+]]:3 = affine.delinearize_index %[[WG_ID_Z]] into (%[[C5]], %[[C8]], %[[C4]])
//      CHECK:   "use"(%[[DELINEARIZE]]#2, %[[DELINEARIZE]]#0, %[[WG_ID_X]], %[[WG_ID_Y]], %[[DELINEARIZE]]#1)

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>]>
hal.executable private @no_loop_do_nothing {
  hal.executable.variant public @no_loop_do_nothing target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @no_loop_do_nothing layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
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
    hal.executable.export public @no_loop_default_workgroup_count layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %0:3 = flow.dispatch.workgroup_count_from_slice %arg1, %arg2
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
