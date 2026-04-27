// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-assign-workgroup-local-memory)))))" --split-input-file --verify-diagnostics %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

// CHECK-LABEL: hal.executable private @single_local_alloc
// CHECK:       hal.executable.export public @dispatch
// CHECK-SAME:    workgroup_local_memory = 1024 : index
hal.executable private @single_local_alloc {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: memref.alloc() {iree_codegen.local_memory_range = array<i64: 0, 1024>}
        %0 = memref.alloc() : memref<16x16xf32, #iree_codegen.workgroup_local>
        memref.dealloc %0 : memref<16x16xf32, #iree_codegen.workgroup_local>
        return
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

// CHECK-LABEL: hal.executable private @packing_and_alignment
// CHECK:       hal.executable.export public @dispatch
// CHECK-SAME:    workgroup_local_memory = 80 : index
hal.executable private @packing_and_alignment {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: memref.alloc() {iree_codegen.local_memory_range = array<i64: 0, 3>}
        %0 = memref.alloc() : memref<3xi4, #iree_codegen.workgroup_local>
        // CHECK: memref.alloc() {iree_codegen.local_memory_range = array<i64: 16, 16>}
        %1 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        // CHECK: memref.alloc() {alignment = 64 : i64, iree_codegen.local_memory_range = array<i64: 64, 16>}
        %2 = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #iree_codegen.workgroup_local>
        return
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

// CHECK-LABEL: hal.executable private @strided_layout
// CHECK:       hal.executable.export public @dispatch
// CHECK-SAME:    workgroup_local_memory = 32 : index
hal.executable private @strided_layout {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: memref.alloc() {iree_codegen.local_memory_range = array<i64: 0, 32>}
        %0 = memref.alloc() : memref<4xf32, strided<[2], offset: 1>, #iree_codegen.workgroup_local>
        return
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

// CHECK-LABEL: hal.executable private @mixed_allocs
// CHECK:       hal.executable.export public @dispatch
// CHECK-SAME:    workgroup_local_memory = 4096 : index
hal.executable private @mixed_allocs {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: memref.alloc() {iree_codegen.local_memory_range = array<i64: 0, 4096>}
        %0 = memref.alloc() : memref<1024xf32, #iree_codegen.workgroup_local>
        // CHECK: memref.alloca() : memref<16xf32>
        %1 = memref.alloca() : memref<16xf32>
        return
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

hal.executable private @dynamic_shape_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch(%size: index) {
        // expected-error @+1 {{workgroup local memory allocations must have static shape}}
        %0 = memref.alloc(%size) : memref<?xf32, #iree_codegen.workgroup_local>
        return
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

hal.executable private @dynamic_layout_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        %c1 = arith.constant 1 : index
        // expected-error @+1 {{workgroup local memory allocations must have static layout}}
        %0 = memref.alloc()[%c1] : memref<4xf32, strided<[?], offset: 0>, #iree_codegen.workgroup_local>
        return
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

hal.executable private @preassigned_alloc_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        // expected-error @+1 {{already has a workgroup local memory assignment}}
        %0 = memref.alloc() {iree_codegen.local_memory_range = array<i64: 0, 16>} : memref<4xf32, #iree_codegen.workgroup_local>
        return
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

hal.executable private @preexisting_export_requirement_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    // expected-error @+1 {{already has a workgroup local memory requirement}}
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {workgroup_local_memory = 16 : index}
    builtin.module {
      func.func @dispatch() {
        %0 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        return
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

hal.executable private @private_helper_alloc_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func private @helper() {
        // expected-error @+1 {{workgroup local memory allocations are only supported in HAL executable exports}}
        %0 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        return
      }
      func.func @dispatch() {
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

hal.executable private @alloca_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        // expected-error @+1 {{workgroup local memory is only supported for memref.alloc}}
        %0 = memref.alloca() : memref<4xf32, #iree_codegen.workgroup_local>
        return
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

hal.executable private @non_entry_alloc_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        cf.br ^bb1
      ^bb1:
        // expected-error @+1 {{workgroup local memory allocations must be in the function entry block}}
        %0 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

hal.executable private @signature_local_memory_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // expected-error @+1 {{workgroup local memory is only supported for memref.alloc results}}
      func.func @dispatch(%arg0: memref<4xf32, #iree_codegen.workgroup_local>) {
        return
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

// CHECK-LABEL: hal.executable private @no_local_allocs
// CHECK:       hal.executable.export public @dispatch
// CHECK-NOT:   workgroup_local_memory
hal.executable private @no_local_allocs {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        %0 = memref.alloca() : memref<16xf32>
        return
      }
    }
  }
}
