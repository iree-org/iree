// RUN: iree-opt -iree-convert-to-nvvm -split-input-file %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
func.func @abs_ex_dispatch_0() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) offset(%c128) : memref<16xf32>
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

// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr<i32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr<f32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr<f32> {llvm.align = 16 : i32})
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %[[PTRI8:.+]] = llvm.bitcast %[[ARG1]] : !llvm.ptr<f32> to !llvm.ptr<i8>
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %[[PTRI8]][%[[C128]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//       CHECK:   %[[PTR:.+]] = llvm.bitcast %[[OFF]] : !llvm.ptr<i8> to !llvm.ptr<f32>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd

// -----

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

// CHECK-LABEL: llvm.func @abs_dynamic
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr<i32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr<f32> {llvm.align = 16 : i32}, %[[ARG2:.+]]: !llvm.ptr<f32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %[[ARG3:.+]]: i32, %[[ARG4:.+]]: i32)
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %{{.*}} = llvm.zext %[[ARG4]] : i32 to i64
//       CHECK:   %[[PTRI8:.+]] = llvm.bitcast %[[ARG1]] : !llvm.ptr<f32> to !llvm.ptr<i8>
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %[[PTRI8]][%[[C128]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//       CHECK:   %[[PTR:.+]] = llvm.bitcast %[[OFF]] : !llvm.ptr<i8> to !llvm.ptr<f32>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd

// -----

// Test that we handle correctly the case where bindings are sparse (set 0
// binding 0 is not used).
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

// CHECK-LABEL: llvm.func @dead_symbol
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr<i32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %[[ARG1:.+]]: !llvm.ptr<f32> {llvm.align = 16 : i32})
//      CHECK:    llvm.fadd

// -----

// A single binding may contain different data types.
// Test that we cast pointers correctly.
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

// CHECK-LABEL: llvm.func @mixed_type
//  CHECK-SAME: (%[[ARG0:.+]]: !llvm.ptr<i32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr<f32> {llvm.align = 16 : i32})
//       CHECK:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
//       CHECK:   %[[PTRI8:.+]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<i32> to !llvm.ptr<i8>
//       CHECK:   %[[OFF:.+]] = llvm.getelementptr %[[PTRI8]][%[[C128]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
//       CHECK:   %[[PTR:.+]] = llvm.bitcast %[[OFF]] : !llvm.ptr<i8> to !llvm.ptr<f32>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %[[PTR]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   nvvm.read.ptx.sreg.tid.x
//       CHECK:   llvm.fadd
  
// -----

func.func @shared_memory_lowering() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %0 = memref.alloc() : memref<1x16x32xf32, 3>
  %1 = memref.alloc() : memref<1x32x16xf32, 3>
  %2 = memref.alloc() : memref<1x8x16xf32, 3> 
  vector.store %cst, %1[%c0, %c0, %c0] : memref<1x32x16xf32, 3>, vector<4xf32>
  vector.store %cst, %2[%c0, %c0, %c0] : memref<1x8x16xf32, 3>, vector<4xf32>
  vector.store %cst, %0[%c0, %c0, %c0] : memref<1x16x32xf32, 3>, vector<4xf32>
  return
}

// CHECK-LABEL: llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-LABEL: llvm.func @shared_memory_lowering() {
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.bitcast %{{.*}} : !llvm.ptr<array<0 x i8>, 3> to !llvm.ptr<array<1 x array<16 x array<32 x f32>>>, 3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2048 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.bitcast %{{.*}} : !llvm.ptr<array<0 x i8>, 3> to !llvm.ptr<array<1 x array<32 x array<16 x f32>>>, 3>
//       CHECK: %{{.*}} = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.mlir.constant(4096 : i64) : i64
//  CHECK-NEXT: %{{.*}} = llvm.getelementptr %{{.*}} : (!llvm.ptr<array<0 x i8>, 3>, i64, i64) -> !llvm.ptr<array<0 x i8>, 3>
//  CHECK-NEXT: %{{.*}} = llvm.bitcast %{{.*}} : !llvm.ptr<array<0 x i8>, 3> to !llvm.ptr<array<1 x array<8 x array<16 x f32>>>, 3>

