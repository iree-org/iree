// RUN: iree-opt -iree-convert-to-nvvm %s | IreeFileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
func @abs_ex_dispatch_0() {
  %c0 = constant 0 : index
  %c128 = constant 128 : index
  %0 = hal.interface.binding.subspan @io::@arg0[%c128] : memref<16xf32>
  %1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<16xi32>
  %2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %6 = muli %3, %4 : index
  %7 = addi %6, %5 : index
  %9 = memref.load %0[%7] : memref<16xf32>
  %10 = memref.load %1[%7] : memref<16xi32>
  %11 = sitofp %10 : i32 to f32
  %12 = addf %9, %11 : f32
  memref.store %12, %2[%7] : memref<16xf32>
  return
}
hal.interface private @io  {
  hal.interface.binding @arg0, set=0, binding=4, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=1, binding=2, type="StorageBuffer", access="Write|Discard"
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
