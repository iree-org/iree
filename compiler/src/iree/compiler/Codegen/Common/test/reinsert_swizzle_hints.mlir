// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-reinsert-swizzle-hints))" \
// RUN:   --split-input-file --mlir-print-local-scope %s | FileCheck %s

func.func @load_1d() -> vector<8xbf16> {
  %alloc = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
    : memref<1024xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %v = vector.load %alloc[%c0]
    : memref<1024xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
  return %v : vector<8xbf16>
}

// CHECK-LABEL: func @load_1d
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1024xbf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:   vector.load %[[HINT]][%{{.+}}]

// -----

func.func @load_2d() -> vector<8xbf16> {
  %alloc = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
    : memref<8x128xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %v = vector.load %alloc[%c0, %c0]
    : memref<8x128xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
  return %v : vector<8xbf16>
}

// CHECK-LABEL: func @load_2d
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<8x128xbf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[COLLAPSED:.+]] = memref.collapse_shape %[[ALLOC]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[COLLAPSED]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:   %[[EXPANDED:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   vector.load %[[EXPANDED]][%{{.+}}, %{{.+}}]

// -----

func.func @store_2d(%v: vector<8xbf16>) {
  %alloc = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
    : memref<8x128xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  vector.store %v, %alloc[%c0, %c0]
    : memref<8x128xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
  return
}

// CHECK-LABEL: func @store_2d
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<8x128xbf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[COLLAPSED:.+]] = memref.collapse_shape %[[ALLOC]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[COLLAPSED]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:   %[[EXPANDED:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   vector.store %{{.+}}, %[[EXPANDED]][%{{.+}}, %{{.+}}]

// -----

func.func @no_hint_without_swizzle() -> vector<8xbf16> {
  %alloc = memref.alloc() : memref<1024xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %v = vector.load %alloc[%c0]
    : memref<1024xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
  return %v : vector<8xbf16>
}

// CHECK-LABEL: func @no_hint_without_swizzle
//       CHECK:   %[[ALLOC:.+]] = memref.alloc()
//       CHECK:   vector.load %[[ALLOC]][%{{.+}}]
//   CHECK-NOT:   swizzle_hint

// -----

func.func @trace_through_subview() -> vector<8xbf16> {
  %alloc = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
    : memref<8x128xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %subview = memref.subview %alloc[0, 0][4, 128][1, 1]
    : memref<8x128xbf16, #gpu.address_space<workgroup>>
    to memref<4x128xbf16, strided<[128, 1]>, #gpu.address_space<workgroup>>
  %v = vector.load %subview[%c0, %c0]
    : memref<4x128xbf16, strided<[128, 1]>, #gpu.address_space<workgroup>>, vector<8xbf16>
  return %v : vector<8xbf16>
}

// CHECK-LABEL: func @trace_through_subview
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<8x128xbf16, #gpu.address_space<workgroup>>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ALLOC]]
//       CHECK:   %[[COLLAPSED:.+]] = memref.collapse_shape %[[SUBVIEW]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   %[[HINT:.+]] = iree_codegen.swizzle_hint %[[COLLAPSED]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:   %[[EXPANDED:.+]] = memref.expand_shape %[[HINT]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   vector.load %[[EXPANDED]][%{{.+}}, %{{.+}}]

// -----

func.func @trace_through_scf_for() -> vector<8xbf16> {
  %alloc = memref.alloc() {iree_codegen.swizzle = #iree_codegen.xor_shuffle<128, 8>}
    : memref<8x128xbf16, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%arg = %alloc) -> memref<8x128xbf16, #gpu.address_space<workgroup>> {
    %v = vector.load %arg[%c0, %c0]
      : memref<8x128xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
    scf.yield %arg : memref<8x128xbf16, #gpu.address_space<workgroup>>
  }
  %epilogue = vector.load %res[%c0, %c0]
    : memref<8x128xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
  return %epilogue : vector<8xbf16>
}

// CHECK-LABEL: func @trace_through_scf_for
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<8x128xbf16, #gpu.address_space<workgroup>>
//       CHECK:   scf.for
//       CHECK:     %[[C1:.+]] = memref.collapse_shape %{{.+}} {{\[\[}}0, 1{{\]\]}}
//       CHECK:     %[[H1:.+]] = iree_codegen.swizzle_hint %[[C1]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:     %[[E1:.+]] = memref.expand_shape %[[H1]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:     vector.load %[[E1]]
//       CHECK:   %[[C2:.+]] = memref.collapse_shape %{{.+}} {{\[\[}}0, 1{{\]\]}}
//       CHECK:   %[[H2:.+]] = iree_codegen.swizzle_hint %[[C2]][#iree_codegen.xor_shuffle<128, 8>]
//       CHECK:   %[[E2:.+]] = memref.expand_shape %[[H2]] {{\[\[}}0, 1{{\]\]}}
//       CHECK:   vector.load %[[E2]]
