// RUN: iree-opt -iree-convert-to-nvvm -split-input-file %s | IreeFileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
func @abs_ex_dispatch_0() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(4) offset(%c128) : memref<16xf32>
  %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : memref<16xi32>
  %2 = hal.interface.binding.subspan type(StorageBuffer) set(1) binding(2) : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
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

func @abs_dynamic() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %s = hal.interface.load.constant offset = 1 : index
  %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(4) offset(%c128) : memref<?xf32>{%s}
  %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) : memref<16xi32>
  %2 = hal.interface.binding.subspan type(StorageBuffer) set(1) binding(2) : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
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
func @dead_symbol() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : memref<16xi32>
  %2 = hal.interface.binding.subspan type(StorageBuffer) set(1) binding(2) : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
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
func @mixed_type() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) offset(%c128) : memref<16xf32>
  %1 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(0) offset(%c0) : memref<16xi32>
  %2 = hal.interface.binding.subspan type(StorageBuffer) set(0) binding(1) : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
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
