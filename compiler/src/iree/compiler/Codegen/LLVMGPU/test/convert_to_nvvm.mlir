// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-nvvm))))" --iree-gpu-test-target=sm_60 --split-input-file %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @abs_ex_dispatch_0() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: ?>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<16xi32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: ?>>
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
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly},
//  CHECK-SAME:  %[[ARG2:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef})
//  CHECK: %[[FADD:.+]] = llvm.fadd %{{.*}}, %{{.*}}  : f32
//  CHECK: %[[ADDR:.+]] = llvm.getelementptr %[[ARG2]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK: llvm.store %[[FADD]], %[[ADDR]] : f32, !llvm.ptr
// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @abs_dynamic {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_dynamic layout(#pipeline_layout)
    builtin.module {
      func.func @abs_dynamic() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c5 = arith.constant 5 : index
        %c7 = arith.constant 7 : index
        %o = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
        %d0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
        %d1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
        %d2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) offset(%o) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>{%d0, %d1, %d2}
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?x?xi32>{%d0, %d1, %d2}
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : memref<?x?x?xf32>{%d0, %d1, %d2}
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
//  CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]: i32 {llvm.noundef},
//  CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]: i32 {llvm.noundef},
//  CHECK-SAME:  %[[ARG5:[a-zA-Z0-9]+]]: i32 {llvm.noundef},
//  CHECK-SAME:  %[[ARG6:[a-zA-Z0-9]+]]: i32 {llvm.noundef})
//   CHECK-DAG:   %[[OFFSET:.+]] = llvm.zext %[[ARG3]] : i32 to i64
//   CHECK-DAG:   %[[D1:.+]] = llvm.zext %[[ARG5]] : i32 to i64
//   CHECK-DAG:   %[[D2:.+]] = llvm.zext %[[ARG6]] : i32 to i64
//   CHECK: %[[GEP1:.+]] = llvm.getelementptr %[[ARG1]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK: %[[GEP:.+]] = llvm.getelementptr %[[GEP1]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK: %[[LOAD:.+]] = llvm.load %[[GEP]] : !llvm.ptr -> f32
//   CHECK: %[[GEP2:.+]] = llvm.getelementptr %[[ARG0]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i32
//   CHECK: llvm.load %[[GEP2]] : !llvm.ptr -> i32
//   CHECK: %[[FADD:.+]] = llvm.fadd %[[LOAD]], %{{.*}}  : f32
//   CHECK: %[[ADD:.+]] = llvm.add
//   CHECK: %[[ADD2:.+]] = llvm.add
//   CHECK: %[[ADDR:.+]] = llvm.getelementptr %[[ARG2]][%[[ADD2]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//   CHECK: llvm.store %[[FADD]], %[[ADDR]] : f32, !llvm.ptr


// -----

// Test that we handle correctly the case where bindings are sparse (set 0
// binding 0 is not used).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @dead_symbol {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @dead_symbol layout(#pipeline_layout)
    builtin.module {
      func.func @dead_symbol() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<16xi32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16xf32>
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
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef})
//      CHECK:    llvm.fadd

// -----

// A single binding may contain different data types.
// Test that we cast pointers correctly.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @mixed_type {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @mixed_type layout(#pipeline_layout)
    builtin.module {
      func.func @mixed_type() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c128) : memref<16xf32, strided<[1], offset: ?>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c0) : memref<16xi32>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: ?>>
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
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef})
//       CHECK:   %[[BYTES_PER_BIT:.+]] = llvm.mlir.constant(8 : i64) : i64
//       CHECK:   %[[BITS_PER_ELEM:.+]] = llvm.mlir.constant(32 : i64) : i64
//       CHECK:   %[[BYTE_OFFSET:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %[[OFFSET_BITS:.+]] = llvm.mul %[[BYTE_OFFSET]], %[[BYTES_PER_BIT]]
//       CHECK:   %[[OFFSET_ELEMS:.+]] = llvm.udiv %[[OFFSET_BITS]], %[[BITS_PER_ELEM]]
//       CHECK:   nvvm.read.ptx.sreg.tid.x
//       CHECK:   llvm.getelementptr %[[ARG0]][%[[OFFSET_ELEMS]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       CHECK:   llvm.fadd

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @shared_memory_lowering {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
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
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(512 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2560 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @shared_memory_dealloc_elision {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @shared_memory_dealloc_elision layout(#pipeline_layout)
    builtin.module {
// CHECK-LABEL: llvm.func @shared_memory_dealloc_elision() {
      func.func @shared_memory_dealloc_elision() {
        %f0 = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        //     CHECK: llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
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

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @shared_memory_lowering_aligned_alloc {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
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
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @check_not_readonly {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @check_not_readonly layout(#pipeline_layout)
    builtin.module {
      func.func @check_not_readonly() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<16xi32>
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: ?>>
        %b11 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) flags(ReadOnly) : memref<16xi32>
        %b12 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) offset(%c128) : memref<16xf32, strided<[1], offset: ?>>
        %b21 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) flags(ReadOnly) : memref<16xi32>
        %b22 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) offset(%c128) flags(ReadOnly) : memref<16xf32, strided<[1], offset: ?>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %0[%7] : memref<16xf32, strided<[1], offset: ?>>
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
//  CHECK-NOT: (%[[ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef, llvm.readonly},

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @complex {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @complex layout(#pipeline_layout)
    builtin.module {
      func.func @complex() {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c128) flags(ReadOnly) : memref<16xcomplex<f32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<16xf32>
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

// -----

// Check that we don't choke on memref of index.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @shared_memory_lowering_index {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @shared_memory_lowering_index layout(#pipeline_layout)
    builtin.module {
      func.func @shared_memory_lowering_index() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0> : vector<4xindex>
        %0 = memref.alloc() : memref<1x16x32xindex, #gpu.address_space<workgroup>>
        vector.store %cst, %0[%c0, %c0, %c0] : memref<1x16x32xindex, #gpu.address_space<workgroup>>, vector<4xindex>
        return
      }
    }
  }
}
//       CHECK: llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-LABEL: llvm.func @shared_memory_lowering_index() {
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @masked_load_store {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @masked_load_store layout(#pipeline_layout)
    builtin.module {
      func.func @masked_load_store() {
        %c0 = arith.constant 0 : index
        %idx = gpu.thread_id x
        %pass_thru = arith.constant dense<0.000000e+00> : vector<1xf32>
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32, #gpu.address_space<global>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<64xf32, #gpu.address_space<global>>
        %mask = vector.create_mask %idx : vector<1xi1>
        %ld = vector.maskedload %0[%idx], %mask, %pass_thru : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32> into vector<1xf32>
        vector.maskedstore %1[%idx], %mask, %ld : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @masked_load_store
//       CHECK:   %[[MASK_BIT:.+]] = llvm.icmp "sgt" {{.*}} : vector<1xi64>
//       CHECK:   llvm.intr.masked.load %{{.*}}, %[[MASK_BIT]]
//       CHECK:   llvm.intr.masked.store %{{.*}}, %[[MASK_BIT]]

// -----
// Test workgroup size lowering
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @interface_wg_size {
  hal.executable.variant @rocm target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @interface_wg_size layout(#pipeline_layout) attributes {
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module attributes {} {
      func.func @interface_wg_size() {
        %c0 = arith.constant 0.0 : f32
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %subspan = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<64x64xf32>
        memref.store %c0, %subspan[%workgroup_size_x, %workgroup_size_y] : memref<64x64xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @interface_wg_size
//       CHECK:   %[[WGDIMX:.+]] = nvvm.read.ptx.sreg.ntid.x
//       CHECK:   %[[WGDIMY:.+]] = nvvm.read.ptx.sreg.ntid.y
