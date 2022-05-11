// RUN: iree-opt --iree-convert-to-rocdl %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
func.func @abs_ex_dispatch_0() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<16xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<16xf32>
  %3 = gpu.block_id x
  %4 = gpu.block_dim x
  %5 = gpu.thread_id x
  %6 = arith.muli %3, %4 : index
  %7 = arith.addi %6, %5 : index
  %9 = memref.load %1[%7] : memref<16xf32>
  %10 = memref.load %2[%7] : memref<16xf32>
  %11 = arith.addf %9, %10 : f32
  memref.store %11, %0[%7] : memref<16xf32>
  return
}

// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%{{.*}}: !llvm.ptr<f32> {llvm.align = 16 : i32}, %{{.*}}: !llvm.ptr<f32> {llvm.align = 16 : i32},
//  CHECK-SAME:  %{{.*}}: !llvm.ptr<f32> {llvm.align = 16 : i32})
//      CHECK:    rocdl.workgroup.dim.x
//      CHECK:    llvm.fadd
