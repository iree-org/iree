// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-tensors -iree-linalg-fusion %s | IreeFileCheck %s

// CHECK-LABEL: @pw_fusion_two
func @pw_fusion_two(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>, %arg2 : tensor<4x8xi32>) -> tensor<4x8xi32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9$._-]+}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: i32
  // CHECK: %[[TEMP:[a-zA-Z0-9$._-]+]] = muli %[[ARG0]], %[[ARG1]]
  // CHECK: addi %[[TEMP]], %[[ARG2]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%4, %arg2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %5 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @pw_fusion_three
func @pw_fusion_three(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>, %arg2 : tensor<4x8xi32>, %arg3: tensor<4x8xi32>) -> tensor<4x8xi32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9$._-]+}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]: i32
  // CHECK: %[[TEMP1:[a-zA-Z0-9$._-]+]] = muli %[[ARG0]], %[[ARG1]]
  // CHECK: %[[TEMP2:[a-zA-Z0-9$._-]+]] = addi %[[TEMP1]], %[[ARG2]]
  // CHECK: subi %[[TEMP2]], %[[ARG3]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%4, %arg2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %6 = "xla_hlo.subtract"(%5, %arg3) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %6: tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @pw_fusion_dag
func @pw_fusion_dag(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>, %arg2 : tensor<4x8xi32>, %arg3: tensor<4x8xi32>) -> tensor<4x8xi32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9$._-]+}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-DAG: %[[TEMP1:[a-zA-Z0-9$._-]+]] = muli %[[ARG0]], %[[ARG1]]
  // CHECK-DAG: %[[TEMP2:[a-zA-Z0-9$._-]+]] = addi %[[ARG2]], %[[ARG3]]
  // CHECK: subi %[[TEMP1]], %[[TEMP2]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%arg2, %arg3) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %6 = "xla_hlo.subtract"(%4, %5) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %6: tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @pw_fusion_dag2
func @pw_fusion_dag2(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>, %arg2 : tensor<4x8xi32>) -> tensor<4x8xi32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9$._-]+}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-SAME: %[[ARG3:[a-zA-Z0-9$._-]+]]: i32
  // CHECK-DAG: %[[TEMP1:[a-zA-Z0-9$._-]+]] = muli %[[ARG0]], %[[ARG1]]
  // CHECK-DAG: %[[TEMP2:[a-zA-Z0-9$._-]+]] = addi %[[ARG2]], %[[ARG3]]
  // CHECK: subi %[[TEMP1]], %[[TEMP2]]
  // CHECK-NOT: linalg.generic
  %3 = "xla_hlo.multiply"(%arg0, %arg1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %4 = "xla_hlo.add"(%arg0, %arg2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.subtract"(%3, %4) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  return %5: tensor<4x8xi32>
}
