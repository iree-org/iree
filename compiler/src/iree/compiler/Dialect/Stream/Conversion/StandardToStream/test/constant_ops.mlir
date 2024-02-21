// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @constantTensor
util.func public @constantTensor() {
  // CHECK: %[[CST:.+]] = stream.tensor.constant : tensor<2xi32> in !stream.resource<constant> = dense<[1, 2]> : tensor<2xi32>
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<constant>
  // CHECK: %[[T:.+]] = stream.async.transfer %[[CST]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  util.return
}

// -----

// CHECK-LABEL: @emptyTensor
util.func public @emptyTensor() {
  // CHECK: %[[CST:.+]] = stream.tensor.constant : tensor<2x0xi32> in !stream.resource<constant> = dense<> : tensor<2x0xi32>
  %0 = arith.constant dense<> : tensor<2x0xi32>
  util.return
}
