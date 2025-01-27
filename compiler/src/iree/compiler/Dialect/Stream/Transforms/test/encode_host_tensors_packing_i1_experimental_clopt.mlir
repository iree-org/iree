// This is only used to test the experimental packing flag. When the default
// is changed the encode_host_tensors.mlir test should be updated and used
// instead and this file should be deleted.

// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors --iree-experimental-packed-i1-storage %s | FileCheck %s

// CHECK-LABEL: @tensorSizeOfUnalignedPackedI1
util.func @tensorSizeOfUnalignedPackedI1() -> index {
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  %0 = stream.tensor.sizeof tensor<12xi1> : index
  // CHECK: return %[[C2]] : index
  util.return %0 : index
}

// -----

// CHECK-LABEL: @tensorSizeOfAlignedPackedI1
util.func @tensorSizeOfAlignedPackedI1() -> index {
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
  %0 = stream.tensor.sizeof tensor<24xi1> : index
  // CHECK: util.return %[[C3]] : index
  util.return %0 : index
}
