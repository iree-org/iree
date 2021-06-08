// RUN: iree-opt -split-input-file -iree-flow-promote-tensor-loads %s | IreeFileCheck %s

func @tensor_extract(%arg0 : tensor<1xi32>, %arg1 : index) -> i32 {
  // CHECK: %[[RESULT:.*]] = flow.tensor.load %arg0[%arg1]
  // CHECK: return %[[RESULT]]
  %extract = tensor.extract %arg0[%arg1] : tensor<1xi32>
  return %extract : i32
}

// -----
func @tensor_extract_i1(%arg0 : tensor<1xi1>, %arg1 : index) -> i1 {
  // CHECK: %[[ZEXT:.*]] = zexti %arg0 : tensor<1xi1> to tensor<1xi8>
  // CHECK: %[[LOADED:.*]] = flow.tensor.load %[[ZEXT]][%arg1] : tensor<1xi8>
  // CHECK: %[[RESULT:.*]] = trunci %[[LOADED]] : i8 to i1
  // CHECK: return %[[RESULT]]
  %extract = tensor.extract %arg0[%arg1] : tensor<1xi1>
  return %extract : i1
}
