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
        %0 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: 32>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: 32>>
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
//       CHECK:   %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   %[[BASE_PTR_INSERT:.+]] = llvm.insertvalue %arg1, %[[UNDEF]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   %[[ALIGNED_PTR_INSERT:.+]] = llvm.insertvalue %arg1, %[[BASE_PTR_INSERT]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   %[[OFFSET:.+]] = llvm.mlir.constant(32 : index) : i64
//       CHECK:   %[[OFFSET_INSERT:.+]] = llvm.insertvalue %[[OFFSET]], %[[ALIGNED_PTR_INSERT]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:    nvvm.read.ptx.sreg.tid.x
//       CHECK:    llvm.fadd

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 4, sets = [
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
        %c3 = arith.constant 3 : index
        %c5 = arith.constant 5 : index
        %c7 = arith.constant 7 : index
        %o = hal.interface.constant.load[0] : index
        %d0 = hal.interface.constant.load[1] : index
        %d1 = hal.interface.constant.load[2] : index
        %d2 = hal.interface.constant.load[3] : index
        %0 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%o) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>{%d0, %d1, %d2}
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?x?xi32>{%d0, %d1, %d2}
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<?x?x?xf32>{%d0, %d1, %d2}
        %9 = memref.load %0[%c3, %c5, %c7] : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>
        %10 = memref.load %1[%c3, %c5, %c7] : memref<?x?x?xi32>
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.addf %9, %11 : f32
        memref.store %12, %2[%c3, %c5, %c7] : memref<?x?x?xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @abs_dynamic
//  CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:  %[[ARG5:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:  %[[ARG6:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:   %[[OFFSET:.+]] = llvm.zext %[[ARG3]] : i32 to i64
//   CHECK-DAG:   %[[D0:.+]] = llvm.zext %[[ARG4]] : i32 to i64
//   CHECK-DAG:   %[[D1:.+]] = llvm.zext %[[ARG5]] : i32 to i64
//   CHECK-DAG:   %[[D2:.+]] = llvm.zext %[[ARG6]] : i32 to i64
//       CHECK:   %[[UNDEF:.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[BASE_PTR_INSERT:.+]] = llvm.insertvalue %[[ARG1]], %[[UNDEF]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[ALIGNED_PTR_INSERT:.+]] = llvm.insertvalue %[[ARG1]], %[[BASE_PTR_INSERT]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[C4:.+]] = llvm.mlir.constant(4 : i64)
//       CHECK:   %[[ELEM_OFFSET:.+]] = llvm.udiv %[[OFFSET]], %[[C4]]
//       CHECK:   %[[OFFSET_INSERT:.+]] = llvm.insertvalue %[[ELEM_OFFSET]], %[[ALIGNED_PTR_INSERT]][2]
//       CHECK:   %[[D0_INSERT:.+]] = llvm.insertvalue %[[D0]], %[[OFFSET_INSERT]][3, 0]
//       CHECK:   %[[D1_INSERT:.+]] = llvm.insertvalue %[[D1]], %[[D0_INSERT]][3, 1]
//       CHECK:   %[[D2_INSERT:.+]] = llvm.insertvalue %[[D2]], %[[D1_INSERT]][3, 2]
//       CHECK:   %[[STRIDE0:.+]] = llvm.mlir.constant(1 : index)
//       CHECK:   %[[STRIDE0_INSERT:.+]] = llvm.insertvalue %[[STRIDE0]], %[[D2_INSERT]][4, 2]
//       CHECK:   %[[D2_EXTRACT:.+]] = llvm.extractvalue %[[STRIDE0_INSERT]][3, 2]
//       CHECK:   %[[STRIDE0_CONSTANT:.+]] = llvm.mlir.constant(1 : i64)
//       CHECK:   %[[STRIDE1:.+]] = llvm.mul %[[STRIDE0_CONSTANT]], %[[D2_EXTRACT]]
//       CHECK:   %[[STRIDE1_INSERT:.+]] = llvm.insertvalue %[[STRIDE1]], %[[STRIDE0_INSERT]][4, 1]
//       CHECK:   %[[D1_EXTRACT:.+]] = llvm.extractvalue %[[STRIDE1_INSERT]][3, 1]
//       CHECK:   %[[STRIDE2:.+]] = llvm.mul %[[STRIDE1]], %[[D1_EXTRACT]]
//       CHECK:   llvm.insertvalue %[[STRIDE2]], %[[STRIDE1_INSERT]][4, 0]
//       CHECK:   llvm.fadd

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
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c128) : memref<16xf32, strided<[1], offset: 4>>
        %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) : memref<16xi32>
        %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: 4>>
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
//       CHECK:   llvm.insertvalue %[[ARG0]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[ARG0]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
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
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: 32>>        
        %b11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) flags(ReadOnly) : memref<16xi32>
        %b12 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c128) : memref<16xf32, strided<[1], offset: 32>>        
        %b21 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) flags(ReadOnly) : memref<16xi32>
        %b22 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: 32>>        
        %2 = hal.interface.binding.subspan set(1) binding(3) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: 32>>
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
hal.executable @complex {
  hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
    hal.executable.export @complex layout(#pipeline_layout)
    builtin.module {
      func.func @complex() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c128) flags(ReadOnly) : memref<16xcomplex<f32>>
        %2 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xcomplex<f32>>
        %10 = complex.re %9 : complex<f32>
        %11 = complex.im %9 : complex<f32>
        %12 = arith.addf %10, %11 : f32
        memref.store %12, %2[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @complex
//   CHECK-NOT: unrealized
//       CHECK: llvm.return

