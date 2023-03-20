// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-nvvm))))" --split-input-file %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @abs_ex_dispatch_0() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32>
        %10 = memref.load %1[%7] : memref<16xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %9, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.readonly},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias})
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %arg1[%[[C128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_dynamic {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @abs_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @abs_dynamic() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %s = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c128) : memref<?xf32>{%s}
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<?xf32>
        %10 = memref.load %1[%7] : memref<16xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %9, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @abs_dynamic
//  CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]: i32, %[[ARG4:[a-zA-Z0-9]+]]: i32)
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %{{.*}} = llvm.zext %[[ARG4]] : i32 to i64
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %arg1[%[[C128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd

// -----

// Test that we handle correctly the case where bindings are sparse (set 0
// binding 0 is not used).
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @dead_symbol {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @dead_symbol layout(#pipeline_layout)
    builtin.module {
      func.func @dead_symbol() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %10 = memref.load %1[%7] : memref<16xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %11, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @dead_symbol
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias})
//      CHECK:    llvm.fadd

// -----

// A single binding may contain different data types.
// Test that we cast pointers correctly.
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @mixed_type {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @mixed_type layout(#pipeline_layout)
    builtin.module {
      func.func @mixed_type() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c128) : memref<16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32>
        %10 = memref.load %1[%7] : memref<16xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %9, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}

// CHECK-LABEL: llvm.func @mixed_type
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias})
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %arg0[%[[C128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[OFF]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   nvvm.read.ptx.sreg.tid.x
//       CHECK:   llvm.fadd

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>
hal.executable @shared_memory_lowering {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @shared_memory_lowering layout(#pipeline_layout)
    builtin.module {
      func.func @shared_memory_lowering() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %0 = memref.alloc() : memref<1x16x32xf32, #gpu.address_space<workgroup>>
        %1 = memref.alloc() : memref<1x32x16xf32, #gpu.address_space<workgroup>>
        %2 = memref.alloc() : memref<1x8x16xf32, #gpu.address_space<workgroup>>
        vector.store %cst, %1[%c0, %c0, %c0] : memref<1x32x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
        vector.store %cst, %2[%c0, %c0, %c0] : memref<1x8x16xf32, #gpu.address_space<workgroup>>, vector<4xf32>
        vector.store %cst, %0[%c0, %c0, %c0] : memref<1x16x32xf32, #gpu.address_space<workgroup>>, vector<4xf32>
        return
      }
    }
  }
}
//       CHECK: llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-LABEL: llvm.func @shared_memory_lowering() {
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2048 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(4096 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>
hal.executable @shared_memory_dealloc_elision {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @shared_memory_dealloc_elision layout(#pipeline_layout)
    builtin.module {
// CHECK-LABEL: llvm.func @shared_memory_dealloc_elision() {
      func.func @shared_memory_dealloc_elision() {
        %f0 = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        //     CHECK: llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
        %0 = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
        memref.store %f0, %0[%c0] : memref<1xf32, #gpu.address_space<workgroup>>
        // CHECK-NOT: free
        memref.dealloc %0 : memref<1xf32, #gpu.address_space<workgroup>>
        return
      }
    }
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>
hal.executable @shared_memory_lowering_aligned_alloc {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @shared_memory_lowering_aligned_alloc layout(#pipeline_layout)
    builtin.module {
      func.func @shared_memory_lowering_aligned_alloc() {
        %c0 = arith.constant 0 : index
        %cst_f32 = arith.constant 0.000000e+00 : f32
        %cst_i8 = arith.constant 0 : i8
        %0 = memref.alloc() : memref<1xi8, #gpu.address_space<workgroup>>
        %1 = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
        memref.store %cst_i8, %0[%c0] : memref<1xi8, #gpu.address_space<workgroup>>
        memref.store %cst_f32, %1[%c0] : memref<32xf32, #gpu.address_space<workgroup>>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-LABEL: llvm.func @shared_memory_lowering_aligned_alloc() {
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(4 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @check_not_readonly {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @check_not_readonly layout(#pipeline_layout)
    builtin.module {
      func.func @check_not_readonly() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<16xi32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32>        
        %b11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) flags(ReadOnly) : memref<16xi32>
        %b12 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c128) : memref<16xf32>        
        %b21 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) flags(ReadOnly) : memref<16xi32>
        %b22 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32>        
        %2 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32>
        %10 = memref.load %1[%7] : memref<16xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %9, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @check_not_readonly
//  CHECK-NOT: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.readonly},
