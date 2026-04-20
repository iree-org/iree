// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-absorb-swizzle-hint-to-alloc))" \
// RUN:   --split-input-file --mlir-print-local-scope %s | FileCheck %s

func.func @absorb_swizzle_hint_to_alloc() {
  %alloc0 = memref.alloc() : memref<8192xbf16, #gpu.address_space<workgroup>>
  %hint0 = iree_codegen.swizzle_hint %alloc0[#iree_codegen.xor_shuffle<128, 8>] : memref<8192xbf16, #gpu.address_space<workgroup>> -> memref<8192xbf16, #gpu.address_space<workgroup>>
  %alloc1 = memref.alloc() : memref<16384xbf16, #gpu.address_space<workgroup>>
  %hint1 = iree_codegen.swizzle_hint %alloc1[#iree_codegen.xor_shuffle<256, 16>] : memref<16384xbf16, #gpu.address_space<workgroup>> -> memref<16384xbf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc0 : memref<8192xbf16, #gpu.address_space<workgroup>>
  memref.dealloc %alloc1 : memref<16384xbf16, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func @absorb_swizzle_hint_to_alloc
//   CHECK-DAG:   memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
//  CHECK-SAME:     : memref<8192xbf16, #gpu.address_space<workgroup>>
//   CHECK-DAG:   memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<256, 16>}
//  CHECK-SAME:     : memref<16384xbf16, #gpu.address_space<workgroup>>
//   CHECK-NOT:   iree_codegen.swizzle_hint
//       CHECK:   return
