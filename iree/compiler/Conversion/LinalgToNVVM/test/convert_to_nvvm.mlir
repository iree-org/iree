// RUN: iree-opt -iree-codegen-convert-to-nvvm %s | IreeFileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
func @abs_ex_dispatch_0() {
  %c0 = constant 0 : index
  %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<16xf32>
  %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<16xf32>
  %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<16xf32>
  %3 = "gpu.block_id"() {dimension = "x"} : () -> index
  %4 = "gpu.block_dim"() {dimension = "x"} : () -> index
  %5 = "gpu.thread_id"() {dimension = "x"} : () -> index
  %6 = muli %3, %4 : index
  %7 = addi %6, %5 : index
  %9 = memref.load %1[%7] : memref<16xf32>
  %10 = memref.load %2[%7] : memref<16xf32>
  %11 = addf %9, %10 : f32
  memref.store %11, %0[%7] : memref<16xf32>
  return
}
hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
}

// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%{{.*}}: !llvm.ptr<f32>, %{{.*}}: !llvm.ptr<f32>, %{{.*}}: !llvm.ptr<f32>)
//      CHECK:    nvvm.read.ptx.sreg.tid.x
//      CHECK:    llvm.fadd
