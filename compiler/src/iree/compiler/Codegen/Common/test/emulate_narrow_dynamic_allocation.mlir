// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type='disable-atomic-rmw=true' %s | FileCheck %s --check-prefixes=CHECK,BYTE
// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type='disable-atomic-rmw=true load-store-bitwidth=32' %s | FileCheck %s --check-prefixes=CHECK,WORD

// A partially filled final storage element must be allocated. Two i1 elements
// require one byte or word; zero, aligned and unaligned extents use the same
// ceiling rule as the converted static memref shape.
// BYTE: #[[$SIZE:.*]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// WORD: #[[$SIZE:.*]] = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK-LABEL: @dynamic_alloca_i1
// CHECK-SAME: %[[N:[^: ]+]]: index,
func.func @dynamic_alloca_i1(%n: index, %index: index, %value: i1) -> i1 {
  // CHECK: %[[COUNT:.*]] = affine.max #[[$SIZE]]()[%[[N]]]
  // BYTE: %[[BUFFER:.*]] = memref.alloca(%[[COUNT]]) alignment = 64 : memref<?xi8>
  // WORD: %[[BUFFER:.*]] = memref.alloca(%[[COUNT]]) alignment = 64 : memref<?xi32>
  // CHECK: memref.store {{.*}}, %[[BUFFER]]
  // CHECK: memref.load %[[BUFFER]]
  // CHECK: return
  %buffer = memref.alloca(%n) alignment = 64 : memref<?xi1>
  memref.store %value, %buffer[%index] : memref<?xi1>
  %result = memref.load %buffer[%index] : memref<?xi1>
  return %result : i1
}

// -----

// BYTE: #[[$SIZE:.*]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// WORD: #[[$SIZE:.*]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-LABEL: @dynamic_alloc_i4
// CHECK-SAME: %[[N:[^: ]+]]: index,
func.func @dynamic_alloc_i4(%n: index, %index: index, %value: i4) -> i4 {
  // CHECK: %[[COUNT:.*]] = affine.max #[[$SIZE]]()[%[[N]]]
  // BYTE: %[[BUFFER:.*]] = memref.alloc(%[[COUNT]]) alignment = 64 : memref<?xi8>
  // WORD: %[[BUFFER:.*]] = memref.alloc(%[[COUNT]]) alignment = 64 : memref<?xi32>
  // CHECK: memref.store {{.*}}, %[[BUFFER]]
  // CHECK: memref.load %[[BUFFER]]
  // CHECK: memref.dealloc %[[BUFFER]]
  // CHECK: return
  %buffer = memref.alloc(%n) alignment = 64 : memref<?xi4>
  memref.store %value, %buffer[%index] : memref<?xi4>
  %result = memref.load %buffer[%index] : memref<?xi4>
  memref.dealloc %buffer : memref<?xi4>
  return %result : i4
}
