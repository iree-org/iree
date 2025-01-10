// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors --iree-experimental-packed-i1-storage %s | FileCheck %s

func.func @unaligned_i1_size() -> index {
  %0 = stream.tensor.sizeof tensor<12xi1> : index
  return %0 : index
}
// CHECK: func @unaligned_i1_size() -> index {
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: return %[[C2]] : index

// -----

func.func @aligned_i1_size() -> index {
  %0 = stream.tensor.sizeof tensor<24xi1> : index
  return %0 : index
}

// CHECK: func @aligned_i1_size() -> index {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: return %[[C3]] : index
