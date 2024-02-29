// RUN: iree-opt --split-input-file --iree-hal-dump-executable-sources %s | FileCheck %s

// Tests dumping executable sources to stdout - it's more common to use files
// but this is much easier to test with lit.

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#device_target_cpu = #hal.device.target<"llvm-cpu", [
  #executable_target_embedded_elf_x86_64
]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#device_target_cpu]}  {

  // CHECK: hal.executable public @ex0
  hal.executable private @ex0 {
    // We expect local outputs with attributes inlined:
    // CHECK-NEXT: hal.executable.variant {{.+}} target(<"llvm-cpu"
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
      hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault>
      } {
      ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
        %c1 = arith.constant 1 : index
        %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
        hal.return %0, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @dispatch0() {
          func.return
        }
      }
    }
  }

  // CHECK: hal.executable private @ex1
  hal.executable private @ex1 {
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
      hal.executable.export public @dispatch1 ordinal(0) layout(#pipeline_layout) attributes {
        translation_info = #iree_codegen.translation_info<CPUDefault>
      } {
      ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
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

}
