// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type %s | FileCheck %s

func.func @memref_i4_to_i8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8xi4>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<8xf32>
  return
}
// CHECK-LABEL: func.func @memref_i4_to_i8
// CHECK:         hal.interface.binding.subspan {{.+}} memref<8xi8>
// CHECK:         hal.interface.binding.subspan {{.+}} memref<8xf32>
