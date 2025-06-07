// RUN: iree-opt --split-input-file --iree-hal-dump-executable-benchmarks %s --verify-diagnostics | FileCheck %s

// Tests dumping executable benchmarks to stdout - it's more common to use files
// but this is much easier to test with lit.

// Ensure devices are copied and made available:
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
// CHECK: util.global private @device
util.global private @device = #hal.device.target<"local", [
  #executable_target_embedded_elf_x86_64
]> : !hal.device

#pipeline_layout_0 = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#pipeline_layout_1 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// Executable should be dumped:
// CHECK: hal.executable private @ex0
hal.executable private @ex0 {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout_0) count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch0() {
        func.return
      }
    }

    hal.executable.export public @dispatch1 ordinal(1) layout(#pipeline_layout_1) count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      %1 = arith.addi %0, %arg1 : index
      hal.return %1, %c1, %c1 : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch1() {
        func.return
      }
    }
  }
}

// ===========================================================================
// @dispatch0 benchmark logic:
// ===========================================================================

// CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch0_512_buffer : !hal.buffer
// CHECK-NEXT: util.initializer {
// CHECK: %[[MEMORY_TYPE:.+]] = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
// CHECK: %[[BUFFER_USAGE:.+]] = hal.buffer_usage<"{{.+}}Dispatch{{.+}}"> : i32
// CHECK: %[[BUFFER:.+]] = hal.allocator.allocate<%{{.+}} : !hal.allocator> affinity(%{{.+}}) type(%[[MEMORY_TYPE]]) usage(%[[BUFFER_USAGE]]) : !hal.buffer{%c768}
// CHECK-NEXT: util.global.store %[[BUFFER]], @ex0_embedded_elf_x86_64_dispatch0_512_buffer : !hal.buffer

// CHECK: util.func public @ex0_embedded_elf_x86_64_dispatch0_512(%arg0: i32)
// CHECK-SAME: attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
// CHECK: %[[BATCH_SIZE:.+]] = arith.index_cast %arg0 : i32 to index

// Create command buffer:
// CHECK: %[[CMD:.+]] = hal.command_buffer.create

// CHECK: %[[BUFFER:.+]] = util.global.load @ex0_embedded_elf_x86_64_dispatch0_512_buffer

// Calculate the workgroup count, which we leave symbolic until after
// translation:
// CHECK: %[[WORKGROUP_X:.+]], %[[WORKGROUP_Y:.+]], %[[WORKGROUP_Z:.+]] =
// CHECK-SAME: hal.executable.calculate_workgroups
// CHECK-SAME:     target(@ex0::@embedded_elf_x86_64::@dispatch0)
// CHECK-SAME:     workload([%c512])

// Get executable and target ordinal (outside of the loop).
// CHECK-DAG: %[[EXECUTABLE:.+]] = hal.executable.lookup device({{.+}}) executable(@ex0) : !hal.executable
// CHECK-DAG: %[[ORDINAL_0:.+]] = hal.executable.export.ordinal target(@ex0::@embedded_elf_x86_64::@dispatch0) : index

// Dispatch up to batch size dispatches:
// CHECK: scf.for %{{.+}} = %c0 to %[[BATCH_SIZE]] step %c1 {
// CHECK-NEXT: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer>
// CHECK-SAME:   target(%[[EXECUTABLE:.+]] : !hal.executable)[%[[ORDINAL_0]]]
// CHECK-SAME:   workgroups([%[[WORKGROUP_X]], %[[WORKGROUP_Y]], %[[WORKGROUP_Z]]])
// CHECK-SAME:   constants([%c100_i32, %c200_i32])
// CHECK-SAME:   bindings([
// CHECK-NEXT:     (%[[BUFFER]] : !hal.buffer)[%c0, %c32],
// CHECK-NEXT:     (%[[BUFFER]] : !hal.buffer)[%c256, %c32],
// CHECK-NEXT:     (%[[BUFFER]] : !hal.buffer)[%c512, %c32]
// CHECK-NEXT:   ])
// CHECK-NEXT: hal.command_buffer.execution_barrier
// CHECK-NEXT: }

// Submit and wait for dispatches to complete:
// CHECK: hal.command_buffer.finalize<%[[CMD]] : !hal.command_buffer>
// CHECK: hal.fence.await

// ===========================================================================
// @dispatch1 benchmark logic (note two deduplicated dispatches):
// ===========================================================================

// CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch1_512x1_buffer : !hal.buffer
// CHECK: util.func public @ex0_embedded_elf_x86_64_dispatch1_512x1(%arg0: i32)
// CHECK:   %[[ORDINAL_1A:.+]] = hal.executable.export.ordinal target(@ex0::@embedded_elf_x86_64::@dispatch1) : index
// CHECK:   hal.command_buffer.dispatch<%{{.+}} : !hal.command_buffer> target({{.+}})[%[[ORDINAL_1A]]]

// CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch1_128x32_buffer : !hal.buffer
// CHECK: util.func public @ex0_embedded_elf_x86_64_dispatch1_128x32(%arg0: i32)
// CHECK:   %[[ORDINAL_1B:.+]] = hal.executable.export.ordinal target(@ex0::@embedded_elf_x86_64::@dispatch1) : index
// CHECK:   hal.command_buffer.dispatch<%{{.+}} : !hal.command_buffer> target({{.+}})[%[[ORDINAL_1B]]]

util.func public @main(%dynamic_arg: i32) -> !stream.timepoint attributes {
  stream.affinity = #hal.device.affinity<@device>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c100_i32 = arith.constant 100 : i32
  %c200_i32 = arith.constant 200 : i32
  %c300_i32 = arith.constant 300 : i32
  %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c128} => !stream.timepoint
  %6 = stream.cmd.execute await(%result_timepoint) => with(%result as %result_capture: !stream.resource<transient>{%c128}) {
    // Dispatches with static and dynamic args.
    stream.cmd.dispatch @ex0::@embedded_elf_x86_64::@dispatch0[%c512](%c100_i32, %c200_i32 : i32, i32) {
      ro %result_capture[%c0 for %c32] : !stream.resource<transient>{%c128},
      rw %result_capture[%c32 for %c32] : !stream.resource<transient>{%c128},
      rw %result_capture[%c64 for %c32] : !stream.resource<transient>{%c128}
    }
    // NOTE: today the dynamic args will prevent us from generating
    // benchmarks. We could handle this better by tracking alignment and such.
    stream.cmd.dispatch @ex0::@embedded_elf_x86_64::@dispatch0[%c512](%c300_i32, %dynamic_arg : i32, i32) {
      ro %result_capture[%c0 for %c32] : !stream.resource<transient>{%c128},
      rw %result_capture[%c32 for %c32] : !stream.resource<transient>{%c128},
      rw %result_capture[%c64 for %c32] : !stream.resource<transient>{%c128}
    }

    // Multiple dispatches to a single entry point.
    // Dispatches are deduplicated and the two 128x32x1 should combine.
    stream.cmd.dispatch @ex0::@embedded_elf_x86_64::@dispatch1[%c512, %c1] {
      ro %result_capture[%c0 for %c64] : !stream.resource<transient>{%c128},
      rw %result_capture[%c64 for %c32] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex0::@embedded_elf_x86_64::@dispatch1[%c128, %c32] {
      ro %result_capture[%c0 for %c64] : !stream.resource<transient>{%c128},
      rw %result_capture[%c64 for %c32] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex0::@embedded_elf_x86_64::@dispatch1[%c128, %c32] {
      ro %result_capture[%c0 for %c64] : !stream.resource<transient>{%c128},
      rw %result_capture[%c64 for %c32] : !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  %39 = stream.resource.dealloca await(%6) => %result : !stream.resource<transient>{%c128} => !stream.timepoint
  util.return %39 : !stream.timepoint
}

// -----
// expected-warning@-2 {{multiple devices in the module}}

// Tests that multiple devices fail today.
// We should be creating one benchmark per executable with only the dispatches
// used by that executable.

#executable_target_embedded_elf_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
util.global private @device_a = #hal.device.target<"local", [
  #executable_target_embedded_elf_aarch64
]> : !hal.device
util.global private @device_b = #hal.device.target<"local", [
  #executable_target_embedded_elf_x86_64
]> : !hal.device

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @ex_0 {
  hal.executable.variant public @variant_a target(#executable_target_embedded_elf_aarch64) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch0() {
        func.return
      }
    }
    hal.executable.export public @dispatch1 ordinal(1) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch1() {
        func.return
      }
    }
  }
  hal.executable.variant public @variant_b target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch0() {
        func.return
      }
    }
    hal.executable.export public @dispatch1 ordinal(1) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch1() {
        func.return
      }
    }
  }
}
hal.executable private @ex_1 {
  hal.executable.variant public @variant_b target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
      hal.return %x, %y, %z : index, index, index
    } attributes {
      translation_info = #iree_codegen.translation_info<pipeline = CPUDefault>
    }
    builtin.module {
      func.func @dispatch0() {
        func.return
      }
    }
  }
}

util.func public @main(%resource_a_arg: !stream.resource<transient>, %resource_b_arg: !stream.resource<transient>) -> (!stream.timepoint, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %tp_a = stream.cmd.execute on(#hal.device.affinity<@device_a>) with(%resource_a_arg as %resource_a: !stream.resource<transient>{%c128}) {
    stream.cmd.dispatch @ex_0::@variant_a::@dispatch0[%c512] {
      rw %resource_a[%c0 for %c32] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex_0::@variant_a::@dispatch1[%c512] {
      rw %resource_a[%c0 for %c64] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex_0::@variant_a::@dispatch1[%c128] {
      rw %resource_a[%c0 for %c64] : !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  %tp_b = stream.cmd.execute on(#hal.device.affinity<@device_b>) with(%resource_b_arg as %resource_b: !stream.resource<transient>{%c128}) {
    stream.cmd.dispatch @ex_0::@variant_a::@dispatch0[%c512] {
      rw %resource_b[%c0 for %c32] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex_0::@variant_a::@dispatch1[%c512] {
      rw %resource_b[%c0 for %c64] : !stream.resource<transient>{%c128}
    }
    stream.cmd.dispatch @ex_0::@variant_b::@dispatch0[%c128] {
      rw %resource_b[%c0 for %c64] : !stream.resource<transient>{%c128}
    }
  } => !stream.timepoint
  util.return %tp_a, %tp_b : !stream.timepoint, !stream.timepoint
}
