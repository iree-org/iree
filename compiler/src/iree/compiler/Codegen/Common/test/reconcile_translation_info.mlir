// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-reconcile-translation-info)))" %s --verify-diagnostics | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
hal.executable private @err_multiple_entry_point {
  // expected-error @+1 {{reconciliation for multiple export ops unsupported}}
  hal.executable.variant public @reconcile_workgroup_size target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout)
    hal.executable.export public @entry_point2 layout(#pipeline_layout)
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>]>]>
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
