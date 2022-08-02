// RUN: iree-opt --split-input-file --iree-hal-dump-executable-benchmarks %s | FileCheck %s

// Tests dumping executable benchmarks to stdout - it's more common to use files
// but this is much easier to test with lit.

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm", "embedded-elf-x86_64">
#device_target_cpu = #hal.device.target<"cpu", {
  executable_targets = [#executable_target_embedded_elf_x86_64_]
}>
#executable_layout_0 = #hal.executable.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#executable_layout_1 = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#device_target_cpu]}  {

  // Executable should be dumped:
  // CHECK: hal.executable private @ex0
  hal.executable private @ex0 {
    hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
      hal.executable.export public @dispatch0 ordinal(0) layout(#executable_layout_0) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault workload_per_wg = [4]>
      } {
      ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):
        %c1 = arith.constant 1 : index
        %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
        hal.return %0, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @dispatch0() {
          func.return
        }
      }

      hal.executable.export public @dispatch1 ordinal(1) layout(#executable_layout_1) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault workload_per_wg = [4]>
      } {
      ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):
        %c1 = arith.constant 1 : index
        %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
        hal.return %0, %c1, %c1 : index, index, index
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

  // CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch0_512x1x1_buffer : !hal.buffer
  // CHECK-NEXT: util.initializer {
  // CHECK: %[[BUFFER:.+]] = hal.allocator.allocate<%{{.+}} : !hal.allocator> type("DeviceVisible|DeviceLocal") usage("{{.+}}Dispatch{{.+}}") : !hal.buffer{%c768}
  // CHECK-NEXT: util.global.store %[[BUFFER]], @ex0_embedded_elf_x86_64_dispatch0_512x1x1_buffer : !hal.buffer

  // CHECK: func.func @ex0_embedded_elf_x86_64_dispatch0_512x1x1(%arg0: i32)
  // CHECK-SAME: attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
  // CHECK: %[[BATCH_SIZE:.+]] = arith.index_cast %arg0 : i32 to index

  // Create command buffer:
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create

  // Setup dispatch constants and bindings:
  // CHECK: hal.command_buffer.push_constants<%[[CMD]] : !hal.command_buffer> layout(%{{.+}} : !hal.executable_layout) offset(0) values([%c0_i32, %c0_i32]) : i32, i32
  // CHECK: %[[BUFFER:.+]] = util.global.load @ex0_embedded_elf_x86_64_dispatch0_512x1x1_buffer
  // CHECK: hal.command_buffer.push_descriptor_set<%[[CMD]] : !hal.command_buffer> layout(%{{.+}} : !hal.executable_layout)[%c0] bindings([
  // CHECK-NEXT:    %c0 = (%[[BUFFER]] : !hal.buffer)[%c0, %c32],
  // CHECK-NEXT:    %c1 = (%[[BUFFER]] : !hal.buffer)[%c256, %c32],
  // CHECK-NEXT:    %c2 = (%[[BUFFER]] : !hal.buffer)[%c512, %c32]
  // CHECK-NEXT:  ])

  // Dispatch up to batch size dispatches:
  // CHECK: scf.for %{{.+}} = %c0 to %[[BATCH_SIZE]] step %c1 {
  // CHECK-NEXT: hal.command_buffer.dispatch.symbol<%[[CMD]] : !hal.command_buffer> target(@ex0::@embedded_elf_x86_64::@dispatch0) workgroups([%c128, %c1, %c1])
  // CHECK-NEXT: hal.command_buffer.execution_barrier
  // CHECK-NEXT: }

  // Submit and wait for dispatches to complete:
  // CHECK: hal.command_buffer.finalize<%[[CMD]] : !hal.command_buffer>
  // CHECK: hal.fence.await

  // ===========================================================================
  // @dispatch1 benchmark logic (note two deduplicated dispatches):
  // ===========================================================================

  // CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch1_512x1x1_buffer : !hal.buffer
  // CHECK: func.func @ex0_embedded_elf_x86_64_dispatch1_512x1x1(%arg0: i32)
  // CHECK:   hal.command_buffer.dispatch.symbol<%{{.+}} : !hal.command_buffer> target(@ex0::@embedded_elf_x86_64::@dispatch1) workgroups([%c128, %c1, %c1])

  // CHECK: util.global private mutable @ex0_embedded_elf_x86_64_dispatch1_128x32x1_buffer : !hal.buffer
  // CHECK: func.func @ex0_embedded_elf_x86_64_dispatch1_128x32x1(%arg0: i32)
  // CHECK:   hal.command_buffer.dispatch.symbol<%{{.+}} : !hal.command_buffer> target(@ex0::@embedded_elf_x86_64::@dispatch1) workgroups([%c32, %c1, %c1])

  func.func private @main() -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<transient>{%c128} => !stream.timepoint
    %6 = stream.cmd.execute await(%result_timepoint) => with(%result as %arg0: !stream.resource<transient>{%c128}) {
      // Dispatch with dynamic args.
      stream.cmd.dispatch @ex0::@dispatch0[%c512, %c1, %c1](%c0_i32, %c1_i32 : i32, i32) {
        ro %arg0[%c0 for %c32] : !stream.resource<transient>{%c128},
        rw %arg0[%c32 for %c32] : !stream.resource<transient>{%c128},
        rw %arg0[%c64 for %c32] : !stream.resource<transient>{%c128}
      } attributes {hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ]}

      // Multiple dispatches to a single entry point.
      // Dispatches are deduplicated and the two 128x32x1 should combine.
      stream.cmd.dispatch @ex0::@dispatch1[%c512, %c1, %c1] {
        ro %arg0[%c0 for %c64] : !stream.resource<transient>{%c128},
        rw %arg0[%c64 for %c32] : !stream.resource<transient>{%c128}
      } attributes {hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>
      ]}
      stream.cmd.dispatch @ex0::@dispatch1[%c128, %c32, %c1] {
        ro %arg0[%c0 for %c64] : !stream.resource<transient>{%c128},
        rw %arg0[%c64 for %c32] : !stream.resource<transient>{%c128}
      } attributes {hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>
      ]}
      stream.cmd.dispatch @ex0::@dispatch1[%c128, %c32, %c1] {
        ro %arg0[%c0 for %c64] : !stream.resource<transient>{%c128},
        rw %arg0[%c64 for %c32] : !stream.resource<transient>{%c128}
      } attributes {hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>
      ]}
    } => !stream.timepoint
    %39 = stream.resource.dealloca await(%6) => %result : !stream.resource<transient>{%c128} => !stream.timepoint
    return %39 : !stream.timepoint
  }
}
