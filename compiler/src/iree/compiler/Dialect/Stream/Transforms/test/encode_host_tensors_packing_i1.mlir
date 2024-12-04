// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors --iree-experimental-packed-i1-storage %s | FileCheck %s

#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i1, i1, i1], layouts = [{i1_packed_storage}]>
func.func @unaligned_i1_size() -> index {
  %0 = stream.tensor.sizeof tensor<12xi1, #encoding> : index
  return %0 : index
}
// CHECK: func @unaligned_i1_size() -> index {
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: return %[[C2]] : index

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i1, i1, i1], layouts = [{i1_packed_storage}]>
func.func @aligned_i1_size() -> index {
  %0 = stream.tensor.sizeof tensor<24xi1, #encoding> : index
  return %0 : index
}

// CHECK: func @aligned_i1_size() -> index {
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: return %[[C3]] : index
