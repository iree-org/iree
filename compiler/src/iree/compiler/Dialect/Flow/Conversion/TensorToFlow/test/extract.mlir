// RUN: iree-opt --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

 util.func public @tensor_extract(%arg0 : tensor<1xi32>, %arg1 : index) -> i32 {
  // CHECK: %[[RESULT:.*]] = flow.tensor.load %arg0[%arg1] : tensor<1xi32>
  // CHECK: util.return %[[RESULT]]
  %extract = tensor.extract %arg0[%arg1] : tensor<1xi32>
  util.return %extract : i32
}

// -----

 util.func public @tensor_extract_i1(%arg0 : tensor<1xi1>, %arg1 : index) -> i1 {
  // CHECK: %[[RESULT:.*]] = flow.tensor.load %arg0[%arg1] : tensor<1xi1>
  // CHECK: util.return %[[RESULT]]
  %extract = tensor.extract %arg0[%arg1] : tensor<1xi1>
  util.return %extract : i1
}
