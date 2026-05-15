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
hal.executable private @single_local_alloc {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: %[[PACKED:.+]] = memref.alloc() : memref<1024xi8, #iree_codegen.workgroup_local>
        // CHECK: %[[C0:.+]] = arith.constant 0 : index
        // CHECK: memref.view %[[PACKED]][%[[C0]]][] : memref<1024xi8, #iree_codegen.workgroup_local> to memref<16x16xf32, #iree_codegen.workgroup_local>
        %0 = memref.alloc() : memref<16x16xf32, #iree_codegen.workgroup_local>
        memref.dealloc %0 : memref<16x16xf32, #iree_codegen.workgroup_local>
        return
      }
      // CHECK: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 1024
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

// CHECK-LABEL: hal.executable private @packing_and_alignment
// CHECK:       hal.executable.export public @dispatch
hal.executable private @packing_and_alignment {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: %[[PACKED:.+]] = memref.alloc() : memref<80xi8, #iree_codegen.workgroup_local>
        // CHECK: %[[C0:.+]] = arith.constant 0 : index
        // CHECK: memref.view %[[PACKED]][%[[C0]]][] : memref<80xi8, #iree_codegen.workgroup_local> to memref<3xi4, #iree_codegen.workgroup_local>
        %0 = memref.alloc() : memref<3xi4, #iree_codegen.workgroup_local>
        // CHECK: %[[C16:.+]] = arith.constant 16 : index
        // CHECK: memref.view %[[PACKED]][%[[C16]]][] : memref<80xi8, #iree_codegen.workgroup_local> to memref<4xf32, #iree_codegen.workgroup_local>
        %1 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        // CHECK: %[[C64:.+]] = arith.constant 64 : index
        // CHECK: memref.view %[[PACKED]][%[[C64]]][] : memref<80xi8, #iree_codegen.workgroup_local> to memref<4xf32, #iree_codegen.workgroup_local>
        %2 = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #iree_codegen.workgroup_local>
        return
      }
      // CHECK: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 80
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

// CHECK-LABEL: hal.executable private @zero_sized_local_alloc
hal.executable private @zero_sized_local_alloc {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: %[[PACKED:.+]] = memref.alloc() : memref<0xi8, #iree_codegen.workgroup_local>
        // CHECK: %[[C0:.+]] = arith.constant 0 : index
        // CHECK: memref.view %[[PACKED]][%[[C0]]][] : memref<0xi8, #iree_codegen.workgroup_local> to memref<0xf32, #iree_codegen.workgroup_local>
        %0 = memref.alloc() : memref<0xf32, #iree_codegen.workgroup_local>
        return
      }
      // CHECK: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 0
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

// CHECK-LABEL: hal.executable private @strided_layout
// CHECK:       hal.executable.export public @dispatch
hal.executable private @strided_layout {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: %[[PACKED:.+]] = memref.alloc() : memref<32xi8, #iree_codegen.workgroup_local>
        // CHECK: %[[C0:.+]] = arith.constant 0 : index
        // CHECK: %[[VIEW:.+]] = memref.view %[[PACKED]][%[[C0]]][] : memref<32xi8, #iree_codegen.workgroup_local> to memref<8xf32, #iree_codegen.workgroup_local>
        // CHECK: memref.reinterpret_cast %[[VIEW]] to offset: [1], sizes: [4], strides: [2] : memref<8xf32, #iree_codegen.workgroup_local> to memref<4xf32, strided<[2], offset: 1>, #iree_codegen.workgroup_local>
        %0 = memref.alloc() : memref<4xf32, strided<[2], offset: 1>, #iree_codegen.workgroup_local>
        return
      }
      // CHECK: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 32
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

// CHECK-LABEL: hal.executable private @mixed_allocs
// CHECK:       hal.executable.export public @dispatch
hal.executable private @mixed_allocs {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      // CHECK-LABEL: func.func @dispatch
      func.func @dispatch() {
        // CHECK: %[[PACKED:.+]] = memref.alloc() : memref<4096xi8, #iree_codegen.workgroup_local>
        // CHECK: %[[C0:.+]] = arith.constant 0 : index
        // CHECK: memref.view %[[PACKED]][%[[C0]]][] : memref<4096xi8, #iree_codegen.workgroup_local> to memref<1024xf32, #iree_codegen.workgroup_local>
        %0 = memref.alloc() : memref<1024xf32, #iree_codegen.workgroup_local>
        // CHECK: memref.alloca() : memref<16xf32>
        %1 = memref.alloca() : memref<16xf32>
        return
      }
      // CHECK: iree_codegen.dispatch_config @dispatch workgroup_local_memory = 4096
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

hal.executable private @dynamic_shape_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch(%size: index) {
        // expected-error @+1 {{workgroup local memory allocations must have static shape}}
        %0 = memref.alloc(%size) : memref<?xf32, #iree_codegen.workgroup_local>
        return
      }
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

hal.executable private @preexisting_config_requirement_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        %0 = memref.alloc() : memref<4xf32, #iree_codegen.workgroup_local>
        return
      }
      // expected-error @+1 {{already has a workgroup local memory requirement}}
      iree_codegen.dispatch_config @dispatch workgroup_local_memory = 16 {
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

hal.executable private @private_helper_alloc_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func private @helper() {
        // expected-error @+1 {{workgroup local memory allocations require an iree_codegen.dispatch_config op}}
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
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-linux-gnu"
}>

hal.executable private @alloca_rejected {
  hal.executable.variant public @embedded target(#executable_target) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @dispatch() {
        // expected-error @+1 {{workgroup local memory is only supported for memref.alloc results}}
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
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n8:16:32:64-S128",
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
