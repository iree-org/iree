// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-erase-dead-alloc-and-stores))" --split-input-file %s | FileCheck %s

func.func @dead_alloc() {
  %0 = memref.alloc() : memref<8x64xf32, 3>
  %1 = memref.subview %0[0, 0] [8, 4] [1, 1] : memref<8x64xf32, 3> to
    memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
  vector.transfer_write %cst_0, %1[%c0, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
  return
}

// CHECK-LABEL:   func.func @dead_alloc
//   CHECK-NOT:     memref.alloc
//   CHECK-NOT:     memref.subview
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     return

// -----

func.func @write_slice_preserves_alloc(
    %dest: !pcf.sref<8x4xf32, #pcf.test_scope>) {
  %alloc = memref.alloc() : memref<8x4xf32>
  pcf.write_slice %alloc into %dest[0, 0] [8, 4] [1, 1]
      : memref<8x4xf32> into !pcf.sref<8x4xf32, #pcf.test_scope>
  return
}

// CHECK-LABEL:   func.func @write_slice_preserves_alloc
//       CHECK:     %[[ALLOC:.+]] = memref.alloc() : memref<8x4xf32>
//       CHECK:     pcf.write_slice %[[ALLOC]] into %{{.+}}[0, 0] [8, 4] [1, 1]
//       CHECK:     return
