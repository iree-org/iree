// RUN: iree-opt -split-input-file -iree-vm-conversion %s | FileCheck %s

module {
  // CHECK-LABEL: vm.func private @load_store
  // CHECK-SAME: (%[[BUFFER:.+]]: !vm.buffer, %[[IDX0:.+]]: i32, %[[IDX1:.+]]: i32) -> f32 {
  func @load_store(%buffer: memref<?xf32>, %idx0: index, %idx1: index) -> f32 {
    // CHECK-NEXT: %[[C4_0:.+]] = vm.const.i32 4
    // CHECK-NEXT: %[[OFFSET0:.+]] = vm.mul.i32 %[[IDX0]], %[[C4_0]] : i32
    // CHECK-NEXT: %[[VALUE:.+]] = vm.buffer.load.f32 %[[BUFFER]][%[[OFFSET0]]] : !vm.buffer -> f32
    %0 = memref.load %buffer[%idx0] : memref<?xf32>
    // CHECK-NEXT: %[[C4_1:.+]] = vm.const.i32 4
    // CHECK-NEXT: %[[OFFSET1:.+]] = vm.mul.i32 %[[IDX1]], %[[C4_1]] : i32
    // CHECK-NEXT: vm.buffer.store.f32 %[[VALUE]], %[[BUFFER]][%[[OFFSET1]]] : f32 -> !vm.buffer
    memref.store %0, %buffer[%idx1] : memref<?xf32>
    // CHECK-NEXT: vm.return %[[VALUE]] : f32
    return %0 : f32
  }
}

// -----

module {
  // CHECK: vm.rodata private @__constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  memref.global "private" constant @__constant : memref<2xf32> = dense<[0.0287729427, 0.0297581609]>
  // CHECK-LABEL: vm.func private @load_global
  // CHECK-SAME: (%[[IDX:.+]]: i32) -> f32 {
  func @load_global(%idx: index) -> f32 {
    // CHECK-NEXT: %[[BUFFER:.+]] = vm.const.ref.rodata @__constant : !vm.buffer
    %0 = memref.get_global @__constant : memref<2xf32>
    // CHECK-NEXT: %[[C4:.+]] = vm.const.i32 4
    // CHECK-NEXT: %[[OFFSET:.+]] = vm.mul.i32 %[[IDX]], %[[C4]] : i32
    // CHECK-NEXT: %[[VALUE:.+]] = vm.buffer.load.f32 %[[BUFFER]][%[[OFFSET]]] : !vm.buffer -> f32
    %1 = memref.load %0[%idx] : memref<2xf32>
    // vm.return %[[VALUE]] : f32
    return %1 : f32
  }
}
