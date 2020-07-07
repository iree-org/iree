// RUN: iree-opt -split-input-file -iree-codegen-decompose-hlo-clamp %s | IreeFileCheck %s

// CHECK-LABEL: func @clamp
// CHECK-SAME: (%[[MIN:.+]]: tensor<4xf32>, %[[INPUT:.+]]: tensor<4xf32>, %[[MAX:.+]]: tensor<4xf32>)
func @clamp(%min: tensor<4xf32>, %value: tensor<4xf32>, %max: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CMP_MIN:.+]] = "mhlo.compare"(%[[MIN]], %[[INPUT]]) {comparison_direction = "LT"}
  // CHECK: %[[SLT_MIN:.+]] = "mhlo.select"(%[[CMP_MIN]], %[[INPUT]], %[[MIN]])
  // CHECK: %[[CMP_MAX:.+]] = "mhlo.compare"(%[[SLT_MIN]], %[[MAX]]) {comparison_direction = "LT"}
  // CHECK: %[[SLT_MAX:.+]] = "mhlo.select"(%[[CMP_MAX]], %[[SLT_MIN]], %[[MAX]])
  // CHECK: return %[[SLT_MAX]]
  %0 = "mhlo.clamp"(%min, %value, %max) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
