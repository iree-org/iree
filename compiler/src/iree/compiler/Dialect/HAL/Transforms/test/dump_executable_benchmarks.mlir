// RUN: iree-opt --split-input-file --iree-hal-dump-executable-benchmarks %s | FileCheck %s

// Tests dumping executable benchmarks to stdout - it's more common to use files
// but this is much easier to test with lit.

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#device_target_cpu = #hal.device.target<"llvm-cpu", {
  executable_targets = [#executable_target_embedded_elf_x86_64]
}>
#pipeline_layout_0 = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#pipeline_layout_1 = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#device_target_cpu]}  {

  // Executable should be dumped:
  // CHECK: hal.executable private @ex0
  hal.executable private @ex0 {
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
      hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout_0) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault>
      } {
      ^bb0(%device: !hal.device, %arg0: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @dispatch0() {
          func.return
        }
      }

      hal.executable.export public @dispatch1 ordinal(1) layout(#pipeline_layout_1) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault>
      } {
      ^bb0(%device: !hal.device, %arg0: index, %arg1: index):
        %c1 = arith.constant 1 : index
        %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
        %1 = arith.addi %0, %arg1 : index
        hal.return %1, %c1, %c1 : index, index, index
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
  // CHECK: %[[BUFFER:.+]] = hal.allocator.allocate<%{{.+}} : !hal.allocator> affinity(%{{.+}}) type("DeviceVisible|DeviceLocal") usage("{{.+}}Dispatch{{.+}}") : !hal.buffer{%c768}
  // CHECK-NEXT: util.global.store %[[BUFFER]], @ex0_embedded_elf_x86_64_dispatch0_512_buffer : !hal.buffer

  // CHECK: util.func public @ex0_embedded_elf_x86_64_dispatch0_512(%arg0: i32)
  // CHECK-SAME: attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
  // CHECK: %[[BATCH_SIZE:.+]] = arith.index_cast %arg0 : i32 to index

  // Create command buffer:
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create

  // Setup dispatch constants and bindings:
  // CHECK: hal.command_buffer.push_constants<%[[CMD]] : !hal.command_buffer> layout(%{{.+}} : !hal.pipeline_layout) offset(0) values([%c100_i32, %c200_i32]) : i32, i32
  // CHECK: %[[BUFFER:.+]] = util.global.load @ex0_embedded_elf_x86_64_dispatch0_512_buffer
  // CHECK: hal.command_buffer.push_descriptor_set<%[[CMD]] : !hal.command_buffer> layout(%{{.+}} : !hal.pipeline_layout)[%c0] bindings([
  // CHECK-NEXT:    %c0 = (%[[BUFFER]] : !hal.buffer)[%c0, %c32],
  // CHECK-NEXT:    %c1 = (%[[BUFFER]] : !hal.buffer)[%c256, %c32],
  // CHECK-NEXT:    %c2 = (%[[BUFFER]] : !hal.buffer)[%c512, %c32]
  // CHECK-NEXT:  ])

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
  // CHECK-NEXT: hal.command_buffer.dispatch<%[[CMD]] : !hal.command_buffer> target(%[[EXECUTABLE:.+]] : !hal.executable)[%[[ORDINAL_0]]] workgroups([%[[WORKGROUP_X]], %[[WORKGROUP_Y]], %[[WORKGROUP_Z]]])
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

  util.func public @main(%dynamic_arg: i32) -> !stream.timepoint {
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
}
