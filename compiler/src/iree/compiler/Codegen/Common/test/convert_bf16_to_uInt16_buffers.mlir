// RUN: iree-opt --split-input-file \
// RUN:   --iree-convert-bf16-to-uint16-buffers %s | FileCheck %s

// CHECK-LABEL: @bf16_conversion
func.func @bf16_conversion() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index

  // CHECK-DAG: %[[BUF0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[BUF1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[BUF2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xi16, #spirv.storage_class<StorageBuffer>>{%c8}
  // CHECK-DAG: %[[LOAD0:.+]] = memref.load %[[BUF0]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  // CHECK: %[[CAST0:.+]] = arith.bitcast %[[LOAD0]] : i16 to bf16
  // CHECK-DAG: %[[LOAD1:.+]] = memref.load %[[BUF1]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  // CHECK: %[[CAST1:.+]] = arith.bitcast %[[LOAD1]] : i16 to bf16
  // CHECK: memref.store %{{.+}}, %[[BUF2]][%arg0] : memref<?xi16, #spirv.storage_class<StorageBuffer>>
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xbf16, #spirv.storage_class<StorageBuffer>>{%c8}
  %3 = gpu.thread_id  x
  %4 = gpu.block_dim  x
  scf.for %arg0 = %3 to %c8 step %4 {
    %5 = memref.load %0[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
    %6 = memref.load %1[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
    %7 = arith.extf %5 : bf16 to f32
    %8 = arith.extf %6 : bf16 to f32
    %9 = arith.addf %7, %8 : f32
    %10 = arith.truncf %9 : f32 to bf16
    memref.store %10, %2[%arg0] : memref<?xbf16, #spirv.storage_class<StorageBuffer>>
  }
  return
}
