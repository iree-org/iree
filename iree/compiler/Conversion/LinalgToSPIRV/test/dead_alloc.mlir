// RUN: iree-opt -iree-codegen-optimize-vector-transfer %s | IreeFileCheck %s

module {
  func @dead_alloc() {
    %0 = memref.alloc() : memref<8x64xf32, 3>
    %1 = memref.subview %0[0, 0] [8, 4] [1, 1] : memref<8x64xf32, 3> to 
      memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
    %c0 = constant 0 : index
    %cst_0 = constant dense<0.000000e+00> : vector<1x4xf32>
    vector.transfer_write %cst_0, %1[%c0, %c0] {masked = [false, false]} : 
      vector<1x4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
    return
  }
}

// CHECK-LABEL:   func @dead_alloc
//   CHECK-NOT:     alloc
//   CHECK-NOT:     memref.subview
//   CHECK-NOT:     vector.transfer_write
//       CHECK:     return
