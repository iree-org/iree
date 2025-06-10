// RUN: iree-opt --split-input-file --iree-hal-conversion --canonicalize -cse %s | FileCheck %s

// Tests an end-to-end simple single-dispatch `dispatch(arg0, arg1) -> result`.

util.global private @device : !hal.device

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">

// CHECK: #[[PIPELINE_LAYOUT_ATTR:.+]] = #hal.pipeline.layout
#pipeline_layout = #hal.pipeline.layout<bindings = [
  // CHECK-SAME: #hal.pipeline.binding<storage_buffer>
  #hal.pipeline.binding<storage_buffer>,
  // CHECK-SAME: #hal.pipeline.binding<storage_buffer>
  #hal.pipeline.binding<storage_buffer>,
  // CHECK-SAME: #hal.pipeline.binding<storage_buffer>
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: hal.executable private @ex
hal.executable private @ex {
  hal.executable.variant public @embedded_elf_aarch64 target(#executable_target_embedded_elf_aarch64) {
    // CHECK: hal.executable.export public @dispatch
    // CHECK-NOT: count
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
}

// CHECK: util.func public @simpleDispatch
// CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view) -> !hal.buffer_view
util.func public @simpleDispatch(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {
  stream.affinity = #hal.device.affinity<@device>
} {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index

  // CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Transfer{{.+}}Dispatch{{.+}}"> : i32
  // CHECK: %[[MEMORY_TYPE:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32

  // CHECK: %[[NULL_FENCE:.+]] = util.null : !hal.fence


  // CHECK: %[[ARG0_BUFFER:.+]] = hal.buffer_view.buffer<%[[ARG0]] : !hal.buffer_view> : !hal.buffer

  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device : !hal.device
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator

  // CHECK: hal.buffer.assert<%[[ARG0_BUFFER]] : !hal.buffer>
  // CHECK-SAME: message("tensor")
  // CHECK-SAME: allocator(%[[ALLOCATOR]] : !hal.allocator)
  // CHECK-SAME: minimum_length(%c16)
  // CHECK-SAME: type(DeviceVisible)
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}Dispatch{{.+}}")
  %arg0_resource = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}

  // CHECK: %[[ARG1_BUFFER:.+]] = hal.buffer_view.buffer<%[[ARG1]] : !hal.buffer_view> : !hal.buffer
  // CHECK: hal.buffer.assert<%[[ARG1_BUFFER]] : !hal.buffer>
  // CHECK-SAME: message("tensor")
  // CHECK-SAME: allocator(%[[ALLOCATOR]] : !hal.allocator)
  // CHECK-SAME: minimum_length(%c16)
  // CHECK-SAME: type(DeviceVisible)
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}Dispatch{{.+}}")
  %arg1_resource = stream.tensor.import %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}

  // CHECK: %[[RESULT_BUFFER:.+]] = hal.allocator.allocate<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME: type(%[[MEMORY_TYPE]])
  // CHECK-SAME: usage(%[[BUFFER_USAGE]])
  // CHECK-SAME: : !hal.buffer{%c16}
  %result_resource = stream.resource.alloc uninitialized : !stream.resource<external>{%c16}

  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-SAME: device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME: categories("Transfer|Dispatch")
  %timepoint = stream.cmd.execute once
      with(%arg0_resource as %arg0_capture: !stream.resource<external>{%c16},
            %arg1_resource as %arg1_capture: !stream.resource<external>{%c16},
            %result_resource as %result_capture: !stream.resource<external>{%c16}) {

    // CHECK-DAG: %{{.+}}, %[[FORMAT_AARCH64:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-aarch64")
    // CHECK-DAG: %{{.+}}, %[[FORMAT_X86_64:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-x86_64")
    // CHECK-DAG: %[[SWITCH1:.+]] = arith.select %[[FORMAT_X86_64]], %c1, %c-1
    // CHECK-DAG: %[[SWITCH0:.+]] = arith.select %[[FORMAT_AARCH64]], %c0, %[[SWITCH1]]
    // CHECK: scf.index_switch %[[SWITCH0]]
    // CHECK: case 0 {
    // CHECK-DAG: %[[EXECUTABLE_0:.+]] = hal.executable.lookup device(%[[DEVICE]] : !hal.device) executable(@ex) : !hal.executable
    // CHECK-DAG: %[[ORDINAL_0:.+]] = hal.executable.export.ordinal target(@ex::@embedded_elf_aarch64::@dispatch) : index
    // CHECK:   hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: target(%[[EXECUTABLE_0]] : !hal.executable)[%[[ORDINAL_0]]]
    // CHECK-SAME: workgroups([%c1, %c1, %c1])
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[ARG0_BUFFER]] : !hal.buffer)[%c0, %c16],
    // CHECK-NEXT:   (%[[ARG1_BUFFER]] : !hal.buffer)[%c0, %c16],
    // CHECK-NEXT:   (%[[RESULT_BUFFER]] : !hal.buffer)[%c0, %c16]
    // CHECK-NEXT: ])
    // CHECK:   scf.yield
    // CHECK: }
    // CHECK: case 1 {
    // CHECK-DAG: %[[EXECUTABLE_1:.+]] = hal.executable.lookup device(%[[DEVICE]] : !hal.device) executable(@ex) : !hal.executable
    // CHECK-DAG: %[[ORDINAL_1:.+]] = hal.executable.export.ordinal target(@ex::@embedded_elf_x86_64::@dispatch) : index
    // CHECK:   hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: target(%[[EXECUTABLE_1]] : !hal.executable)[%[[ORDINAL_1]]]
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[ARG0_BUFFER]] : !hal.buffer)[%c0, %c16]
    // CHECK-NEXT:   (%[[ARG1_BUFFER]] : !hal.buffer)[%c0, %c16],
    // CHECK-NEXT:   (%[[RESULT_BUFFER]] : !hal.buffer)[%c0, %c16]
    // CHECK-NEXT: ])
    // CHECK:   scf.yield
    // CHECK: }
    stream.cmd.dispatch {
      @ex::@embedded_elf_aarch64::@dispatch,
      @ex::@embedded_elf_x86_64::@dispatch
    }[%c4, %c1, %c1] {
      ro %arg0_capture[%c0 for %c16] : !stream.resource<external>{%c16},
      ro %arg1_capture[%c0 for %c16] : !stream.resource<external>{%c16},
      wo %result_capture[%c0 for %c16] : !stream.resource<external>{%c16}
    }

  // CHECK: hal.command_buffer.execution_barrier<%[[CMD]] : !hal.command_buffer>
  // CHECK-SAME: source("Dispatch|Transfer|CommandRetire")
  // CHECK-SAME: target("CommandIssue|Dispatch|Transfer")
  // CHECK: hal.command_buffer.finalize<%[[CMD]] : !hal.command_buffer>
  } => !stream.timepoint

  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: hal.device.queue.execute<%[[DEVICE]]
  // CHECK-SAME: wait(%[[NULL_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: commands(%[[CMD]])

  // CHECK: hal.fence.await until([%[[SIGNAL_FENCE]]])
  %result_ready = stream.timepoint.await %timepoint => %result_resource : !stream.resource<external>{%c16}

  // CHECK-DAG: %[[ELEMENT_TYPE:.+]] = hal.element_type<f32>
  // CHECK-DAG: %[[ENCODING_TYPE:.+]] = hal.encoding_type<dense_row_major>
  // CHECK: %[[RESULT_VIEW:.+]] = hal.buffer_view.create
  // CHECK-SAME: buffer(%[[RESULT_BUFFER]] : !hal.buffer)
  // CHECK-SAME: shape([%c4])
  // CHECK-SAME: type(%[[ELEMENT_TYPE]])
  // CHECK-SAME: encoding(%[[ENCODING_TYPE]])
  %result_view = stream.tensor.export %result_ready : tensor<4xf32> in !stream.resource<external>{%c16} -> !hal.buffer_view
  // CHECK: util.return
  util.return %result_view : !hal.buffer_view
}
