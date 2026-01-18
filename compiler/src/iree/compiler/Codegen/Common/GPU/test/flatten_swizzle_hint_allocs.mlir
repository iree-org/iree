// RUN: iree-opt --allow-unregistered-dialect --pass-pipeline="builtin.module(func.func(iree-codegen-flatten-swizzle-hint-allocs))" \
// RUN:   --mlir-print-local-scope %s | FileCheck %s

// Test: 1D alloc should NOT be flattened (already 1D).
func.func @skip_1d_alloc() {
  %alloc = memref.alloc() : memref<2048xf32, #gpu.address_space<workgroup>>
  %0 = iree_codegen.swizzle_hint %alloc[#iree_codegen.rotate_rows<64, 4>] : memref<2048xf32, #gpu.address_space<workgroup>>
  "test.use"(%0) : (memref<2048xf32, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @skip_1d_alloc
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<2048xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.rotate_rows<64, 4>] : memref<2048xf32, #gpu.address_space<workgroup>>
//   CHECK-NOT:   memref.expand_shape
//       CHECK:   "test.use"(%[[HINT]])

// Test: 2D alloc with swizzle hint should be flattened to 1D.
func.func @flatten_2d_alloc() {
  %alloc = memref.alloc() : memref<32x64xf32, #gpu.address_space<workgroup>>
  %0 = iree_codegen.swizzle_hint %alloc[#iree_codegen.rotate_rows<64, 4>] : memref<32x64xf32, #gpu.address_space<workgroup>>
  "test.use"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @flatten_2d_alloc
//       CHECK:   %[[ALLOC1D:.+]] = memref.alloc() : memref<2048xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC1D]][#iree_codegen.rotate_rows<64, 4>] : memref<2048xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[EXPAND:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1{{\]\]}} output_shape [32, 64] : memref<2048xf32, #gpu.address_space<workgroup>> into memref<32x64xf32, #gpu.address_space<workgroup>>
//       CHECK:   "test.use"(%[[EXPAND]])
//   CHECK-NOT:   memref.alloc() : memref<32x64xf32
//   CHECK-NOT:   iree_codegen.swizzle_hint {{.*}} : memref<32x64xf32
//       CHECK:   return

// Test: 3D alloc with swizzle hint should be flattened to 1D.
func.func @flatten_3d_alloc() {
  %alloc = memref.alloc() : memref<4x8x16xf32, #gpu.address_space<workgroup>>
  %0 = iree_codegen.swizzle_hint %alloc[#iree_codegen.rotate_rows<64, 4>] : memref<4x8x16xf32, #gpu.address_space<workgroup>>
  "test.use"(%0) : (memref<4x8x16xf32, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @flatten_3d_alloc
//       CHECK:   %[[ALLOC1D:.+]] = memref.alloc() : memref<512xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC1D]][#iree_codegen.rotate_rows<64, 4>] : memref<512xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[EXPAND:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1, 2{{\]\]}} output_shape [4, 8, 16] : memref<512xf32, #gpu.address_space<workgroup>> into memref<4x8x16xf32, #gpu.address_space<workgroup>>
//       CHECK:   "test.use"(%[[EXPAND]])
//   CHECK-NOT:   memref.alloc() : memref<4x8x16xf32
//   CHECK-NOT:   iree_codegen.swizzle_hint {{.*}} : memref<4x8x16xf32
//       CHECK:   return

// Test: Non-alloc operand should NOT be affected.
func.func @skip_non_alloc(%arg0: memref<32x64xf32, #gpu.address_space<workgroup>>) {
  %0 = iree_codegen.swizzle_hint %arg0[#iree_codegen.rotate_rows<64, 4>] : memref<32x64xf32, #gpu.address_space<workgroup>>
  "test.use"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @skip_non_alloc
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: memref<32x64xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ARG0]][#iree_codegen.rotate_rows<64, 4>] : memref<32x64xf32, #gpu.address_space<workgroup>>
//   CHECK-NOT:   memref.expand_shape
//       CHECK:   "test.use"(%[[HINT]])

// Test: Alloc with multiple uses should NOT be flattened.
func.func @skip_multi_use_alloc() {
  %alloc = memref.alloc() : memref<32x64xf32, #gpu.address_space<workgroup>>
  %0 = iree_codegen.swizzle_hint %alloc[#iree_codegen.rotate_rows<64, 4>] : memref<32x64xf32, #gpu.address_space<workgroup>>
  "test.use"(%alloc) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
  "test.use"(%0) : (memref<32x64xf32, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @skip_multi_use_alloc
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.rotate_rows<64, 4>] : memref<32x64xf32, #gpu.address_space<workgroup>>
//   CHECK-NOT:   memref.expand_shape
//       CHECK:   "test.use"(%[[ALLOC]])
//       CHECK:   "test.use"(%[[HINT]])

// Test: XOR shuffle swizzle attribute.
func.func @flatten_xor_shuffle() {
  %alloc = memref.alloc() : memref<16x128xi8, #gpu.address_space<workgroup>>
  %0 = iree_codegen.swizzle_hint %alloc[#iree_codegen.xor_shuffle<128, 16>] : memref<16x128xi8, #gpu.address_space<workgroup>>
  "test.use"(%0) : (memref<16x128xi8, #gpu.address_space<workgroup>>) -> ()
  return
}

// CHECK-LABEL: func @flatten_xor_shuffle
//       CHECK:   %[[ALLOC1D:.+]] = memref.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC1D]][#iree_codegen.xor_shuffle<128, 16>] : memref<2048xi8, #gpu.address_space<workgroup>>
//       CHECK:   %[[EXPAND:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1{{\]\]}} output_shape [16, 128] : memref<2048xi8, #gpu.address_space<workgroup>> into memref<16x128xi8, #gpu.address_space<workgroup>>
//       CHECK:   "test.use"(%[[EXPAND]])
//   CHECK-NOT:   memref.alloc() : memref<16x128xi8
//   CHECK-NOT:   iree_codegen.swizzle_hint {{.*}} : memref<16x128xi8
//       CHECK:   return
