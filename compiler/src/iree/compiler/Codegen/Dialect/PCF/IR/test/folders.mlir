// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @write_slice_fold_offsets
func.func @write_slice_fold_offsets(%arg0: tensor<8x8xf32>, %dest: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: pcf.write_slice %arg0 into %arg1[10, 20] [8, 8] [1, 1]
  pcf.write_slice %arg0 into %dest[%c10, %c20] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
  return
}

// -----

// CHECK-LABEL: @write_slice_fold_strides
func.func @write_slice_fold_strides(%arg0: tensor<8x8xf32>, %dest: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: pcf.write_slice %arg0 into %arg1[0, 0] [8, 8] [2, 3]
  pcf.write_slice %arg0 into %dest[0, 0] [8, 8] [%c2, %c3] : tensor<8x8xf32> into !pcf.sref<64x128xf32, sync(#pcf.sequential)>
  return
}

// -----

// CHECK-LABEL: @write_slice_fold_mixed
func.func @write_slice_fold_mixed(%arg0: tensor<8x8xf32>, %dest: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  // CHECK: pcf.write_slice %arg0 into %arg1[5, 10] [8, 8] [1, 2]
  pcf.write_slice %arg0 into %dest[%c5, %c10] [8, 8] [1, %c2] : tensor<8x8xf32> into !pcf.sref<64x128xf32, sync(#pcf.sequential)>
  return
}

// -----

// CHECK-LABEL: @write_slice_no_fold_dynamic
func.func @write_slice_no_fold_dynamic(%arg0: tensor<8x8xf32>, %dest: !pcf.sref<32x64xf32, sync(#pcf.sequential)>, %offset: index) {
  // CHECK: pcf.write_slice %arg0 into %arg1[%arg2, 0] [8, 8] [1, 1]
  pcf.write_slice %arg0 into %dest[%offset, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
  return
}

// -----

// CHECK-LABEL: @read_slice_fold_offsets
func.func @read_slice_fold_offsets(%source: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) -> tensor<8x8xf32> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: pcf.read_slice %arg0[10, 20] [8, 8] [1, 1]
  %0 = pcf.read_slice %source[%c10, %c20] [8, 8] [1, 1] : !pcf.sref<32x64xf32, sync(#pcf.sequential)> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: @read_slice_fold_strides
func.func @read_slice_fold_strides(%source: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) -> tensor<8x8xf32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: pcf.read_slice %arg0[0, 0] [8, 8] [2, 3]
  %0 = pcf.read_slice %source[0, 0] [8, 8] [%c2, %c3] : !pcf.sref<64x128xf32, sync(#pcf.sequential)> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: @read_slice_fold_mixed
func.func @read_slice_fold_mixed(%source: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) -> tensor<8x8xf32> {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  // CHECK: pcf.read_slice %arg0[5, 10] [8, 8] [1, 2]
  %0 = pcf.read_slice %source[%c5, %c10] [8, 8] [1, %c2] : !pcf.sref<64x128xf32, sync(#pcf.sequential)> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: @read_slice_no_fold_dynamic
func.func @read_slice_no_fold_dynamic(%source: !pcf.sref<32x64xf32, sync(#pcf.sequential)>, %offset: index) -> tensor<8x8xf32> {
  // CHECK: pcf.read_slice %arg0[%arg1, 0] [8, 8] [1, 1]
  %0 = pcf.read_slice %source[%offset, 0] [8, 8] [1, 1] : !pcf.sref<32x64xf32, sync(#pcf.sequential)> to tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: @get_memref_fold_offsets
func.func @get_memref_fold_offsets(%source: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) -> memref<8x8xf32, strided<[?, ?], offset: ?>> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // CHECK: pcf.get_memref %arg0[10, 20] [8, 8] [1, 1]
  %0 = pcf.get_memref %source[%c10, %c20] [8, 8] [1, 1] : !pcf.sref<32x64xf32, sync(#pcf.sequential)> to memref<8x8xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<8x8xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: @get_memref_fold_strides
func.func @get_memref_fold_strides(%source: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) -> memref<8x8xf32, strided<[?, ?], offset: ?>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: pcf.get_memref %arg0[0, 0] [8, 8] [2, 3]
  %0 = pcf.get_memref %source[0, 0] [8, 8] [%c2, %c3] : !pcf.sref<64x128xf32, sync(#pcf.sequential)> to memref<8x8xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<8x8xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: @get_memref_fold_mixed
func.func @get_memref_fold_mixed(%source: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) -> memref<8x8xf32, strided<[?, ?], offset: ?>> {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  // CHECK: pcf.get_memref %arg0[5, 10] [8, 8] [1, 2]
  %0 = pcf.get_memref %source[%c5, %c10] [8, 8] [1, %c2] : !pcf.sref<64x128xf32, sync(#pcf.sequential)> to memref<8x8xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<8x8xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: @get_memref_no_fold_dynamic
func.func @get_memref_no_fold_dynamic(%source: !pcf.sref<32x64xf32, sync(#pcf.sequential)>, %offset: index) -> memref<8x8xf32, strided<[?, ?], offset: ?>> {
  // CHECK: pcf.get_memref %arg0[%arg1, 0] [8, 8] [1, 1]
  %0 = pcf.get_memref %source[%offset, 0] [8, 8] [1, 1] : !pcf.sref<32x64xf32, sync(#pcf.sequential)> to memref<8x8xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<8x8xf32, strided<[?, ?], offset: ?>>
}
