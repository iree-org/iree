// RUN: iree-opt --split-input-file --iree-hal-materialize-resource-caches %s | FileCheck %s

//      CHECK: util.global private @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %descriptor_set_layout = hal.descriptor_set_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     flags("None")
// CHECK-SAME:     bindings([
// CHECK-SAME:       #hal.descriptor_set.binding<0, storage_buffer>,
// CHECK-SAME:       #hal.descriptor_set.binding<1, storage_buffer>
// CHECK-SAME:     ]) : !hal.descriptor_set_layout
// CHECK-NEXT:   util.global.store %descriptor_set_layout, @_descriptor_set_layout_0 : !hal.descriptor_set_layout

// CHECK-LABEL: @descriptorSetLayoutLookup
func.func @descriptorSetLayoutLookup(%device : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK-NEXT: %[[LAYOUT:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup device(%device : !hal.device)
                                        flags("None")
                                        bindings([
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]) : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

// CHECK: util.global private @_descriptor_set_layout_0 : !hal.descriptor_set_layout

//      CHECK: util.global private @_pipeline_layout_0 : !hal.pipeline_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[SET0:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %pipeline_layout = hal.pipeline_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     push_constants(1)
// CHECK-SAME:     layouts([%[[SET0]]]) : !hal.pipeline_layout
// CHECK-NEXT:   util.global.store %pipeline_layout, @_pipeline_layout_0 : !hal.pipeline_layout

// CHECK-LABEL: @exeLayoutLookup
func.func @exeLayoutLookup(%device : !hal.device) -> !hal.pipeline_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_pipeline_layout_0 : !hal.pipeline_layout
  %0 = hal.pipeline_layout.lookup device(%device : !hal.device)
                                    layout(#hal.pipeline.layout<push_constants = 1, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>
  ]>) : !hal.pipeline_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.pipeline_layout
}

// -----

// CHECK: util.global private @_descriptor_set_layout_0
// CHECK: util.global private @_descriptor_set_layout_1

//      CHECK: util.global private @_pipeline_layout_0 : !hal.pipeline_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[SET0:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %[[SET1:.+]] = util.global.load @_descriptor_set_layout_1 : !hal.descriptor_set_layout
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %pipeline_layout = hal.pipeline_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     push_constants(1)
// CHECK-SAME:     layouts([%[[SET0]], %[[SET1]]]) : !hal.pipeline_layout
// CHECK-NEXT:   util.global.store %pipeline_layout, @_pipeline_layout_0 : !hal.pipeline_layout

// CHECK-LABEL: @sharedLayoutLookup
func.func @sharedLayoutLookup(%device : !hal.device) -> !hal.pipeline_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_pipeline_layout_0 : !hal.pipeline_layout
  %0 = hal.pipeline_layout.lookup device(%device : !hal.device)
                                    layout(#hal.pipeline.layout<push_constants = 1, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>,
    #hal.descriptor_set.layout<1, bindings = [
      #hal.descriptor_set.binding<0, uniform_buffer>,
      #hal.descriptor_set.binding<1, uniform_buffer>
    ]>
  ]>) : !hal.pipeline_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.pipeline_layout
}

// CHECK: @otherDescriptorSetLayoutLookup
func.func @otherDescriptorSetLayoutLookup(%device : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup device(%device : !hal.device)
                                        flags(None)
                                        bindings([
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]) : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

#pipeline_layout_0 = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#pipeline_layout_1 = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#hal.device.target<"llvm-cpu">]} {

// TODO(scotttodd): Test without depending on a specific HAL target? Or move to HAL/Target/*/test/?
//   - If there is no matching hal.executable.variant then the executable will not be cached
hal.executable @exe {
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @entry0 ordinal(0) layout(#pipeline_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.export @entry0_alias ordinal(0) layout(#pipeline_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.export @entry1 ordinal(1) layout(#pipeline_layout_1) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.constant.block() -> (i32, i32) as ("foo", "bar") {
      %c123 = arith.constant 123 : i32
      %c456 = arith.constant 456 : i32
      hal.return %c123, %c456 : i32, i32
    }
    hal.executable.constant.block(%device: !hal.device) -> i32 as "baz" {
      %ok, %query = hal.device.query<%device : !hal.device> key("sys" :: "baz") : i1, i32
      cf.cond_br %ok, ^bb_ok, ^bb_fail
    ^bb_ok:
      hal.return %query : i32
    ^bb_fail:
      %dummy = arith.constant 0 : i32
      hal.return %dummy : i32
    }
  }
}

// CHECK-DAG: util.global private @_descriptor_set_layout_0
// CHECK-DAG: util.global private @_pipeline_layout_0
// CHECK-DAG: util.global private @_descriptor_set_layout_1
// CHECK-DAG: util.global private @_pipeline_layout_1

// CHECK: util.global private @_executable_exe : !hal.executable
// CHECK-NEXT: util.initializer {
// CHECK:   %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
// CHECK:   %[[RET:.+]] = hal.device.switch<%[[DEVICE]] : !hal.device> -> !hal.executable
// CHECK:   #hal.device.match.executable.format<"vmvx-bytecode-fb"> {

// Dependent layouts:
// CHECK:     %[[LAYOUT0:.+]] = util.global.load @_pipeline_layout_0 : !hal.pipeline_layout
// CHECK:     %[[LAYOUT0_2:.+]] = util.global.load @_pipeline_layout_0 : !hal.pipeline_layout
// CHECK:     %[[LAYOUT1:.+]] = util.global.load @_pipeline_layout_1 : !hal.pipeline_layout

// Constant block initializers:
// CHECK:     %[[CONST_01:.+]]:2 = func.call @__constant_block_0()
// CHECK:     %[[CONST_2:.+]] = func.call @__constant_block_1(%[[DEVICE]])

// Executable creation:
// CHECK:     %[[EXE:.+]] = hal.executable.create
// CHECK-SAME:  device(%[[DEVICE]] : !hal.device)
// CHECK-SAME:  target(@exe::@vmvx)
// CHECK-SAME:  layouts([%[[LAYOUT0]], %[[LAYOUT0_2]], %[[LAYOUT1]]])
// CHECK-SAME:  constants([%[[CONST_01]]#0, %[[CONST_01]]#1, %[[CONST_2]]])
// CHECK-SAME:  : !hal.executable

// CHECK:     hal.return %[[EXE]] : !hal.executable
// CHECK:   },
// CHECK:   #hal.match.always {
// CHECK:     %[[NULL:.+]] = util.null : !hal.executable
// CHECK:     hal.return %[[NULL]] : !hal.executable
// CHECK:   }
// CHECK:   util.global.store %[[RET]], @_executable_exe : !hal.executable

// Inlined constant block functions (here we ensure all blocks are cloned):
// CHECK: func.func private @__constant_block_0() -> (i32, i32)
// CHECK-DAG: %[[C0:.+]] = arith.constant 123
// CHECK-DAG: %[[C1:.+]] = arith.constant 456
// CHECK: return %[[C0]], %[[C1]]
// CHECK: func.func private @__constant_block_1(%[[BLOCK_DEVICE:.+]]: !hal.device) -> i32
// CHECK:   %[[OK:.+]], %[[VALUE:.+]] = hal.device.query<%[[BLOCK_DEVICE]] : !hal.device> key("sys" :: "baz")
// CHECK:   cf.cond_br %[[OK]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   return %[[VALUE]]
// CHECK: ^bb2:
// CHECK:   %[[DUMMY:.+]] = arith.constant 0
// CHECK:   return %[[DUMMY]]

// CHECK-LABEL: @exeLookup
func.func @exeLookup(%device : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable_exe : !hal.executable
  %0 = hal.executable.lookup device(%device : !hal.device)
                             executable(@exe) : !hal.executable
  // CHECK-NEXT: return %[[EXE]]
  return %0 : !hal.executable
}

}

// -----

// Tests that materialization no-ops when resource caches have already been
// materialized. Today this is rather simplistic and just bails if the names
// match with the expectation being that users are mostly just running through
// with --compile-to=hal and not trying to mutate intermediate HAL state. We
// could rework the pass to support only materializing what's required based on
// what resources are looked up.

#pipeline_layout_0 = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#hal.device.target<"llvm-cpu">]} {

util.global private @_descriptor_set_layout_0 : !hal.descriptor_set_layout
util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %descriptor_set_layout = hal.descriptor_set_layout.create device(%device : !hal.device) flags("None") bindings([#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>]) : !hal.descriptor_set_layout
  util.global.store %descriptor_set_layout, @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  util.initializer.return
}

util.global private @_pipeline_layout_0 : !hal.pipeline_layout
util.initializer {
  %_descriptor_set_layout_0 = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %device = hal.ex.shared_device : !hal.device
  %pipeline_layout = hal.pipeline_layout.create device(%device : !hal.device) push_constants(0) layouts([%_descriptor_set_layout_0]) : !hal.pipeline_layout
  util.global.store %pipeline_layout, @_pipeline_layout_0 : !hal.pipeline_layout
  util.initializer.return
}

util.global private @_executable_exe : !hal.executable
util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %0 = hal.device.switch<%device : !hal.device> -> !hal.executable
  #hal.device.match.executable.format<"vmvx-bytecode-fb"> {
    %_pipeline_layout_0 = util.global.load @_pipeline_layout_0 : !hal.pipeline_layout
    %exe = hal.executable.create device(%device : !hal.device) target(@exe0::@vmvx) layouts([%_pipeline_layout_0]) : !hal.executable
    hal.return %exe : !hal.executable
  },
  #hal.match.always {
    %1 = util.null : !hal.executable
    hal.return %1 : !hal.executable
  }
  util.global.store %0, @_executable_exe : !hal.executable
  util.initializer.return
}

hal.executable @exe {
  hal.executable.variant @vmvx target(<"vmvx", "vmvx-bytecode-fb">) {
    hal.executable.export @entry ordinal(0) layout(#pipeline_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-LABEL: @exeLookup
func.func @exeLookup(%device : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable_exe : !hal.executable
  %0 = util.global.load @_executable_exe : !hal.executable
  // CHECK-NEXT: return %[[EXE]]
  return %0 : !hal.executable
}

}
