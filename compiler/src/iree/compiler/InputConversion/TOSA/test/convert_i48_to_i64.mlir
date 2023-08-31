// RUN: iree-opt --split-input-file --iree-convert-i48-to-i64 --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @test_all_i48_converted
func.func @test_all_i48_converted(%arg0: tensor<2x2xi48>) -> tensor<2x2xi48> {
  // CHECK: %[[ADD:.+]] = tosa.add %arg0, %arg0 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  // CHECK: %[[SUB:.+]] = tosa.sub %[[ADD]], %arg0 : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  // CHECK: return %[[SUB]] : tensor<2x2xi64>
  %0 = tosa.add %arg0, %arg0 : (tensor<2x2xi48>, tensor<2x2xi48>) -> tensor<2x2xi48>
  %1 = tosa.sub %0, %arg0 : (tensor<2x2xi48>, tensor<2x2xi48>) -> tensor<2x2xi48>
  return %1 : tensor<2x2xi48>
}

// CHECK-LABEL: @test_other_types_not_converted
func.func @test_other_types_not_converted(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: %[[ADD:.+]] = tosa.add %arg0, %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: %[[SUB:.+]] = tosa.sub %[[ADD]], %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: return %[[SUB]] : tensor<2x2xi32>
  %0 = tosa.add %arg0, %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  %1 = tosa.sub %0, %arg0 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: @test_attrs_converted
func.func @test_attrs_converted() -> (i48, tensor<2xi48>) {
  // CHECK: %[[ARITH_C:.+]] = arith.constant 0 : i64
  // CHECK: %[[TOSA_C:.+]] = "tosa.const"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK: return %[[ARITH_C]], %[[TOSA_C]] : i64, tensor<2xi64>
  %0 = "arith.constant"() {value = 0 : i48} : () -> i48
  %1 = "tosa.const"() <{value = dense<0> : tensor<2xi48>}> : () -> tensor<2xi48>
  return %0, %1 : i48, tensor<2xi48>
}
