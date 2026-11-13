// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-move-stack-allocations-to-workgroup-local-memory)))))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-move-stack-allocations-to-workgroup-local-memory, iree-llvmcpu-assign-workgroup-local-memory, iree-llvmcpu-check-ir-before-llvm-conversion)))))" --split-input-file %s | FileCheck %s --check-prefix=PIPELINE

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

// Allocations within the 32768 byte default limit stay on the stack.

// CHECK-LABEL: hal.executable private @within_limit
// PIPELINE-LABEL: hal.executable private @within_limit
hal.executable private @within_limit {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK:     memref.alloca() alignment = 64 : memref<4096xf32>
        // CHECK:     memref.alloca() : memref<4096xi32>
        // CHECK-NOT: memref.alloc()
        %0 = memref.alloca() alignment = 64 : memref<4096xf32>
        %1 = memref.alloca() : memref<4096xi32>
        return
      }
      // CHECK:     iree_codegen.dispatch_config @dispatch
      // CHECK-NOT: workgroup_local_memory
      iree_codegen.dispatch_config @dispatch {
        %c1 = arith.constant 1 : index
        iree_codegen.yield %c1, %c1, %c1 : index, index, index
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

// 65536 + 16384 + 64 bytes exceed the limit. Moving the largest allocation is
// enough, the other two stay on the stack.

// CHECK-LABEL: hal.executable private @move_largest
// PIPELINE-LABEL: hal.executable private @move_largest
hal.executable private @move_largest {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      // CHECK-SAME:    %[[INDEX:[a-zA-Z0-9]+]]: index
      func.func @dispatch(%index: index) -> f32 {
        // CHECK:     %[[SMALL:.+]] = memref.alloca() : memref<16xi32>
        // CHECK:     %[[LOCAL:.+]] = memref.alloc() alignment = 64 : memref<16384xf32, #iree_codegen.workgroup_local>
        // CHECK:     %[[CAST:.+]] = memref.memory_space_cast %[[LOCAL]] : memref<16384xf32, #iree_codegen.workgroup_local> to memref<16384xf32>
        // CHECK:     %[[MEDIUM:.+]] = memref.alloca() : memref<4096xi32>
        // CHECK:     memref.load %[[CAST]][%[[INDEX]]] : memref<16384xf32>
        // PIPELINE:  %[[PACKED:.+]] = memref.alloc() : memref<65536xi8, #iree_codegen.workgroup_local>
        // PIPELINE:  memref.view %[[PACKED]]
        %small = memref.alloca() : memref<16xi32>
        %large = memref.alloca() alignment = 64 : memref<16384xf32>
        %medium = memref.alloca() : memref<4096xi32>
        %value = memref.load %large[%index] : memref<16384xf32>
        return %value : f32
      }
      // PIPELINE: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 65536
      iree_codegen.dispatch_config @dispatch {
        %c1 = arith.constant 1 : index
        iree_codegen.yield %c1, %c1, %c1 : index, index, index
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  max_stack_allocation_size = 65536 : i64,
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

// Two allocations of 40000 bytes each exceed the configured limit of 65536
// bytes; one of them is moved.

// CHECK-LABEL: hal.executable private @configured_limit
// PIPELINE-LABEL: hal.executable private @configured_limit
hal.executable private @configured_limit {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK:      memref.alloc() : memref<10000xi32, #iree_codegen.workgroup_local>
        // CHECK:      memref.alloca() : memref<10000xf32>
        // CHECK-NOT:  memref.alloc()
        %0 = memref.alloca() : memref<10000xi32>
        %1 = memref.alloca() : memref<10000xf32>
        return
      }
      // PIPELINE: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 40000
      iree_codegen.dispatch_config @dispatch {
        %c1 = arith.constant 1 : index
        iree_codegen.yield %c1, %c1, %c1 : index, index, index
      }
    }
  }
}
