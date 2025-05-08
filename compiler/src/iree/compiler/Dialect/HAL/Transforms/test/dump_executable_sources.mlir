// RUN: iree-opt --split-input-file --iree-hal-dump-executable-sources %s | FileCheck %s

// Tests dumping executable sources to stdout - it's more common to use files
// but this is much easier to test with lit.

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: hal.executable public @ex0
hal.executable private @ex0 {
  // We expect local outputs with attributes inlined:
  // CHECK-NEXT: hal.executable.variant {{.+}} target(<"llvm-cpu"
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
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

// CHECK: hal.executable private @ex1
hal.executable private @ex1 {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
    hal.executable.export public @dispatch1 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
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
