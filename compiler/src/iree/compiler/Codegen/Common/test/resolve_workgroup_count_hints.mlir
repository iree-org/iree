// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-resolve-workgroup-count-hints, canonicalize, cse)))" \
// RUN:   %s --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @resolve_multiple_calls {
  hal.executable.variant public @resolve_multiple_calls target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        func.call @fn1() : () -> ()
        func.call @fn2() : () -> ()
        return
      }
      func.func @fn1() {
        iree_codegen.workgroup_count_hint(1, 5, 3)
        return
      }
      func.func @fn2() {
        iree_codegen.workgroup_count_hint(4, 2, 6)
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @resolve_multiple_calls
//       CHECK:   hal.executable.export public @entry_point
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device) -> (index, index, index)
//       CHECK:     hal.return %c4, %c5, %c6 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @resolve_multiple_hints {
  hal.executable.variant public @resolve_multiple_calls target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        iree_codegen.workgroup_count_hint(%o0)
        iree_codegen.workgroup_count_hint(%o1)
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @resolve_multiple_hints
//       CHECK:   hal.executable.export public @entry_point
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index) -> (index, index, index)
//       CHECK:     %[[MAX:.+]] = arith.maxsi %[[ARG1]], %[[ARG2]] : index
//       CHECK:     hal.return %[[MAX]], %c1, %c1 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @resolve_multiple_conditioned_hints {
  hal.executable.variant public @resolve_multiple_calls target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %b = arith.cmpi slt, %o0, %o1 : index
        scf.if %b {
          iree_codegen.workgroup_count_hint(1, 2, 3)
        } else {
          iree_codegen.workgroup_count_hint(4, 5, 6)
        }
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @resolve_multiple_conditioned_hints
//       CHECK:   hal.executable.export public @entry_point
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index) -> (index, index, index)
//       CHECK:     %[[CMP1:.+]] = arith.cmpi slt, %[[ARG1]], %[[ARG2]] : index
//       CHECK:     %[[SEL1:.+]] = arith.select %[[CMP1]], %c1, %c0 : index
//       CHECK:     %[[SEL2:.+]] = arith.select %[[CMP1]], %c2, %c0 : index
//       CHECK:     %[[SEL3:.+]] = arith.select %[[CMP1]], %c3, %c0 : index
//       CHECK:     %[[CMP2:.+]] = arith.cmpi sge, %[[ARG1]], %[[ARG2]] : index
//       CHECK:     %[[SEL4:.+]] = arith.select %[[CMP2]], %c4, %c0 : index
//       CHECK:     %[[SEL5:.+]] = arith.select %[[CMP2]], %c5, %c0 : index
//       CHECK:     %[[SEL6:.+]] = arith.select %[[CMP2]], %c6, %c0 : index
//       CHECK:     %[[MAX1:.+]] = arith.maxsi %[[SEL1]], %[[SEL4]] : index
//       CHECK:     %[[MAX2:.+]] = arith.maxsi %[[SEL2]], %[[SEL5]] : index
//       CHECK:     %[[MAX3:.+]] = arith.maxsi %[[SEL3]], %[[SEL6]] : index
//       CHECK:     hal.return %[[MAX1]], %[[MAX2]], %[[MAX3]] : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @two_entry_points_shared_function {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @entry_point2 layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point1() {
        %c3 = arith.constant 3 : index
        func.call @shared_fn(%c3) : (index) -> ()
        return
      }
      func.func @entry_point2() {
        %c7 = arith.constant 7 : index
        func.call @shared_fn(%c7) : (index) -> ()
        return
      }
      func.func @shared_fn(%arg0: index) {
        iree_codegen.workgroup_count_hint(%arg0)
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @two_entry_points_shared_function
//       CHECK:   hal.executable.export public @entry_point1
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device) -> (index, index, index)
//       CHECK:     hal.return %c3, %c1, %c1 : index, index, index
//       CHECK:   hal.executable.export public @entry_point2
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device) -> (index, index, index)
//       CHECK:     hal.return %c7, %c1, %c1 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @multiple_entry_points_two_hints_shared_fn {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @entry_point2 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point1() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %c5 = arith.constant 5 : index
        %cmp = arith.cmpi slt, %o0, %c5 : index
        scf.if %cmp {
          func.call @helper_fn(%o0) : (index) -> ()
        }
        return
      }
      func.func @entry_point2() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %c2 = arith.constant 2 : index
        %mul = arith.muli %o0, %c2 : index
        func.call @helper_fn(%mul) : (index) -> ()
        return
      }
      func.func @helper_fn(%arg0: index) {
        iree_codegen.workgroup_count_hint(%arg0, 1, 1)
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @multiple_entry_points_two_hints_shared_fn
//       CHECK:   hal.executable.export public @entry_point1
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG1:.+]]: index) -> (index, index, index)
//       CHECK:     %[[CMP:.+]] = arith.cmpi slt, %[[ARG1]], %c5 : index
//       CHECK:     %[[SEL1:.+]] = arith.select %[[CMP]], %[[ARG1]], %c0 : index
//       CHECK:     %[[SEL2:.+]] = arith.select %[[CMP]], %c1, %c0 : index
//       CHECK:     hal.return %[[SEL1]], %[[SEL2]], %[[SEL2]] : index, index, index
//       CHECK:   hal.executable.export public @entry_point2
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG2:.+]]: index) -> (index, index, index)
//       CHECK:     %[[MUL:.+]] = arith.muli %[[ARG2]], %c2 : index
//       CHECK:     hal.return %[[MUL]], %c1, %c1 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @function_chain_increment {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        func.call @increment_once(%o0) : (index) -> ()
        return
      }
      func.func @increment_once(%arg0: index) {
        %c1 = arith.constant 1 : index
        %add = arith.addi %arg0, %c1 : index
        func.call @increment_twice(%add) : (index) -> ()
        return
      }
      func.func @increment_twice(%arg0: index) {
        %c1 = arith.constant 1 : index
        %add = arith.addi %arg0, %c1 : index
        iree_codegen.workgroup_count_hint(%add, 2, 3)
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @function_chain_increment
//       CHECK:   hal.executable.export public @entry_point
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG:.+]]: index) -> (index, index, index)
//       CHECK:     %[[ADD:.+]] = arith.addi %[[ARG]], %c2 : index
//       CHECK:     hal.return %[[ADD]], %c2, %c3 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @nested_scf_if {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %c10 = arith.constant 10 : index
        %c20 = arith.constant 20 : index
        %cmp1 = arith.cmpi slt, %o0, %c10 : index
        scf.if %cmp1 {
          %cmp2 = arith.cmpi slt, %o1, %c20 : index
          scf.if %cmp2 {
            iree_codegen.workgroup_count_hint(%o0, %o1, 5)
          }
        }
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @nested_scf_if
//       CHECK:   hal.executable.export public @entry_point
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index) -> (index, index, index)
//       CHECK:     %[[CMP1:.+]] = arith.cmpi slt, %[[ARG2]], %c20 : index
//       CHECK:     %[[CMP2:.+]] = arith.cmpi slt, %[[ARG1]], %c10 : index
//       CHECK:     %[[AND:.+]] = arith.andi %[[CMP1]], %[[CMP2]] : i1
//       CHECK:     %[[SEL1:.+]] = arith.select %[[AND]], %[[ARG1]], %c0 : index
//       CHECK:     %[[SEL2:.+]] = arith.select %[[AND]], %[[ARG2]], %c0 : index
//       CHECK:     %[[SEL3:.+]] = arith.select %[[AND]], %c5, %c0 : index
//       CHECK:     hal.return %[[SEL1]], %[[SEL2]], %[[SEL3]] : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @optional_i1_param {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point1 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @entry_point2 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @entry_point3 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point1() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %cmp = arith.cmpi slt, %o0, %o1 : index
        func.call @shared_fn_with_i1(%cmp, %o0, %o1) : (i1, index, index) -> ()
        return
      }
      func.func @entry_point2() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        %c0 = arith.constant 0 : index
        %binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<i1>
        %loaded = memref.load %binding[] : memref<i1>
        func.call @shared_fn_with_i1(%loaded, %o0, %o1) : (i1, index, index) -> ()
        return
      }
      func.func @entry_point3() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %b = arith.index_cast %cst1 : index to i1
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        func.call @shared_fn_with_i1(%b, %o0, %o0) : (i1, index, index) -> ()
        return
      }
      func.func @shared_fn_with_i1(%arg0: i1, %arg1: index, %arg2: index) {
        scf.if %arg0 {
          iree_codegen.workgroup_count_hint(%arg1, %arg2, 1)
        } else {
          iree_codegen.workgroup_count_hint(%arg2, %arg1, 2)
        }
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @optional_i1_param
//       CHECK:   hal.executable.export public @entry_point1
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index) -> (index, index, index)
//       CHECK:     %[[CMP1:.+]] = arith.cmpi slt, %[[ARG1]], %[[ARG2]] : index
//       CHECK:     %[[SEL1:.+]] = arith.select %[[CMP1]], %[[ARG1]], %c0 : index
//       CHECK:     %[[SEL2:.+]] = arith.select %[[CMP1]], %[[ARG2]], %c0 : index
//       CHECK:     %[[SEL3:.+]] = arith.select %[[CMP1]], %c1, %c0 : index
//       CHECK:     %[[CMP2:.+]] = arith.cmpi sge, %[[ARG1]], %[[ARG2]] : index
//       CHECK:     %[[SEL4:.+]] = arith.select %[[CMP2]], %[[ARG2]], %c0 : index
//       CHECK:     %[[SEL5:.+]] = arith.select %[[CMP2]], %[[ARG1]], %c0 : index
//       CHECK:     %[[SEL6:.+]] = arith.select %[[CMP2]], %c2, %c0 : index
//       CHECK:     %[[MAX1:.+]] = arith.maxsi %[[SEL1]], %[[SEL4]] : index
//       CHECK:     %[[MAX2:.+]] = arith.maxsi %[[SEL2]], %[[SEL5]] : index
//       CHECK:     %[[MAX3:.+]] = arith.maxsi %[[SEL3]], %[[SEL6]] : index
//       CHECK:     hal.return %[[MAX1]], %[[MAX2]], %[[MAX3]] : index, index, index
//       CHECK:   hal.executable.export public @entry_point2
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index) -> (index, index, index)
//       CHECK:     %[[MAX:.+]] = arith.maxsi %[[ARG3]], %[[ARG4]] : index
//       CHECK:     hal.return %[[MAX]], %[[MAX]], %c2 : index, index, index
//       CHECK:   hal.executable.export public @entry_point3
//  CHECK-SAME:     layout({{.+}}) count(%{{.+}}: !hal.device, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index) -> (index, index, index)
//       CHECK:     hal.return %[[ARG3]], %[[ARG3]], %c2 : index, index, index

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @recursive_function_error {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        func.call @recursive_fn(%o0) : (index) -> ()
        return
      }
      // expected-error @+1 {{detected recursive call when resolving workgroup count hints}}
      func.func @recursive_fn(%arg0: index) {
        func.call @recursive_fn(%arg0) : (index) -> ()
        iree_codegen.workgroup_count_hint(%arg0, 1, 1)
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ordinal_out_of_bounds_error {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %cst0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %cst1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %o0 = iree_tensor_ext.dispatch.workload.ordinal %cst0, 0 : index
        // expected-error @+1 {{ordinal number is higher than the number of workloads captured in the workgroup count region}}
        %o1 = iree_tensor_ext.dispatch.workload.ordinal %cst1, 1 : index
        // expected-error @+1 {{failed to materialize operand slice}}
        iree_codegen.workgroup_count_hint(%o0, %o1, 1)
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @hint_operand_not_ordinal_error {
  hal.executable.variant public @variant target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        %c0 = arith.constant 0 : index
        %binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<index>
        %loaded = memref.load %binding[] : memref<index>
        // expected-error @+1 {{failed to resolve workgroup count hint in terms of workload ordinals}}
        iree_codegen.workgroup_count_hint(%loaded, 1, 1)
        return
      }
    }
  }
}
