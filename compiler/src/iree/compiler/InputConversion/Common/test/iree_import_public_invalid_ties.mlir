// RUN: not iree-opt --split-input-file --iree-import-public %s 2>&1 | FileCheck %s

func.func private @tied_callee(%arg: tensor<4xi32>)
    -> (tensor<4xi32> {iree.abi.tied = 0 : i64}) {
  return %arg : tensor<4xi32>
}

// An explicit mapping that conflicts with the callee ABI is rejected.
func.func @conflicting_caller(%arg: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: tied operands mismatch
  %result = func.call @tied_callee(%arg) {iree.abi.tied_operands = [-1 : index]} : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// A malformed present attribute must not be treated as an omitted mapping.
func.func private @tied_callee(%arg: tensor<4xi32>)
    -> (tensor<4xi32> {iree.abi.tied = 0 : i64}) {
  return %arg : tensor<4xi32>
}
func.func @malformed_caller(%arg: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: iree.abi.tied_operands must be an array
  %result = func.call @tied_callee(%arg) {iree.abi.tied_operands = "bad"} : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
