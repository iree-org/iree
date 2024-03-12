// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-capture-executable-sources{stage=configured})' %s | FileCheck %s

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

// CHECK-DAG: #[[EX0_VARIANT0_LOC:.+]] = loc("module_ex0_variant0.configured.mlir"
// CHECK-DAG: #[[EX1_VARIANT1_LOC:.+]] = loc("module_ex1_variant1.configured.mlir"

// CHECK: hal.executable private @ex0
hal.executable private @ex0 {
  // CHECK-NEXT: hal.executable.variant public @variant0
  // CHECK-SAME: sources({module_ex0_variant0.configured.mlir = dense_resource<module_ex0_variant0.configured.mlir
  hal.executable.variant public @variant0 target(#executable_target) {
    // CHECK: hal.executable.export public @dispatch0
    // CHECK-SAME: source_locs = {configured = #[[EX0_VARIANT0_LOC]]}
    hal.executable.export public @dispatch0 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<CPUDefault>
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
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
  // CHECK-NEXT: hal.executable.variant public @variant1
  // CHECK-SAME: sources({module_ex1_variant1.configured.mlir = dense_resource<module_ex1_variant1.configured.mlir
  hal.executable.variant public @variant1 target(#executable_target) {
    // CHECK: hal.executable.export public @dispatch1
    // CHECK-SAME: source_locs = {configured = #[[EX1_VARIANT1_LOC]]}
    hal.executable.export public @dispatch1 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<CPUDefault>
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @dispatch1() {
        func.return
      }
    }
  }
}

//      CHECK: {-#
// CHECK-NEXT:   dialect_resources: {
// CHECK-NEXT:     builtin: {
// CHECK-NEXT:       module_ex0_variant0.configured.mlir:
// CHECK-NEXT:       module_ex1_variant1.configured.mlir:
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #-}
