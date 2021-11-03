// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @constantTensor
func @constantTensor() {
  // CHECK: %[[CST:.+]] = stream.tensor.constant : tensor<2xi32> in !stream.resource<constant> = dense<[1, 2]> : tensor<2xi32>
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<constant>
  // CHECK: %[[T:.+]] = stream.async.transfer %[[CST]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  return
}
