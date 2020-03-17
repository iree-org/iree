// RUN: iree-opt -split-input-file -iree-hlo-to-linalg-on-tensors -iree-linalg-fusion %s | IreeFileCheck %s

// CHECK-LABEL: @pw_fusion_two
func @pw_fusion_two(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>, %arg2 : memref<4x8xi32>, %arg3: memref<4x8xi32>)
attributes { iree.executable.export, iree.executable.workgroup_size = dense<[32, 8, 1]> : tensor<3xi32>} {
  %0 = iree.load_input(%arg0 : memref<4x8xi32>) : tensor<4x8xi32>
  %1 = iree.load_input(%arg1 : memref<4x8xi32>) : tensor<4x8xi32>
  %2 = iree.load_input(%arg2 : memref<4x8xi32>) : tensor<4x8xi32>
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9_]*}}
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: i32
  // CHECK: [[TEMP:%[a-zA-Z0-9_]*]] = muli [[ARG0]], [[ARG1]]
  // CHECK: addi [[TEMP]], [[ARG2]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%4, %2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  iree.store_output(%5 : tensor<4x8xi32>, %arg3 : memref<4x8xi32>)
  return
}

// -----

// CHECK-LABEL: @pw_fusion_three
func @pw_fusion_three(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>, %arg2 : memref<4x8xi32>, %arg3: memref<4x8xi32>, %arg4: memref<4x8xi32>)
attributes { iree.executable.export, iree.executable.workgroup_size = dense<[32, 8, 1]> : tensor<3xi32>} {
  %0 = iree.load_input(%arg0 : memref<4x8xi32>) : tensor<4x8xi32>
  %1 = iree.load_input(%arg1 : memref<4x8xi32>) : tensor<4x8xi32>
  %2 = iree.load_input(%arg2 : memref<4x8xi32>) : tensor<4x8xi32>
  %3 = iree.load_input(%arg3 : memref<4x8xi32>) : tensor<4x8xi32>
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9_]*}}
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]: i32
  // CHECK: [[TEMP1:%[a-zA-Z0-9_]*]] = muli [[ARG0]], [[ARG1]]
  // CHECK: [[TEMP2:%[a-zA-Z0-9_]*]] = addi [[TEMP1]], [[ARG2]]
  // CHECK: subi [[TEMP2]], [[ARG3]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%4, %2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %6 = "xla_hlo.subtract"(%5, %3) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  iree.store_output(%6 : tensor<4x8xi32>, %arg4 : memref<4x8xi32>)
  return
}

// -----

// CHECK-LABEL: @pw_fusion_dag
func @pw_fusion_dag(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>, %arg2 : memref<4x8xi32>, %arg3: memref<4x8xi32>, %arg4: memref<4x8xi32>)
attributes { iree.executable.export, iree.executable.workgroup_size = dense<[32, 8, 1]> : tensor<3xi32>} {
  %0 = iree.load_input(%arg0 : memref<4x8xi32>) : tensor<4x8xi32>
  %1 = iree.load_input(%arg1 : memref<4x8xi32>) : tensor<4x8xi32>
  %2 = iree.load_input(%arg2 : memref<4x8xi32>) : tensor<4x8xi32>
  %3 = iree.load_input(%arg3 : memref<4x8xi32>) : tensor<4x8xi32>
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9_]*}}
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]: i32
  // CHECK-DAG: [[TEMP1:%[a-zA-Z0-9_]*]] = muli [[ARG0]], [[ARG1]]
  // CHECK-DAG: [[TEMP2:%[a-zA-Z0-9_]*]] = addi [[ARG2]], [[ARG3]]
  // CHECK: subi [[TEMP1]], [[TEMP2]]
  // CHECK-NOT: linalg.generic
  %4 = "xla_hlo.multiply"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.add"(%2, %3) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %6 = "xla_hlo.subtract"(%4, %5) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  iree.store_output(%6 : tensor<4x8xi32>, %arg4 : memref<4x8xi32>)
  return
}

// -----

// CHECK-LABEL: @pw_fusion_dag2
func @pw_fusion_dag2(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>, %arg2 : memref<4x8xi32>, %arg3: memref<4x8xi32>)
attributes { iree.executable.export, iree.executable.workgroup_size = dense<[32, 8, 1]> : tensor<3xi32>} {
  %0 = iree.load_input(%arg0 : memref<4x8xi32>) : tensor<4x8xi32>
  %1 = iree.load_input(%arg1 : memref<4x8xi32>) : tensor<4x8xi32>
  %2 = iree.load_input(%arg2 : memref<4x8xi32>) : tensor<4x8xi32>
  // CHECK: linalg.generic
  // CHECK: ^{{[a-zA-Z0-9_]*}}
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]: i32
  // CHECK-DAG: [[TEMP1:%[a-zA-Z0-9_]*]] = muli [[ARG0]], [[ARG1]]
  // CHECK-DAG: [[TEMP2:%[a-zA-Z0-9_]*]] = addi [[ARG2]], [[ARG3]]
  // CHECK: subi [[TEMP1]], [[TEMP2]]
  // CHECK-NOT: linalg.generic
  %3 = "xla_hlo.multiply"(%0, %1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %4 = "xla_hlo.add"(%0, %2) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  %5 = "xla_hlo.subtract"(%3, %4) :  (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
  iree.store_output(%5 : tensor<4x8xi32>, %arg3 : memref<4x8xi32>)
  return
}
