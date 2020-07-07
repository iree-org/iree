// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s
flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
    workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<4x4xf32> {
      %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<4x3xf32>, tensor<3x4xf32>) -> tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0
// CHECK-DAG:   hal.executable.entry_point @simpleMath_rgn_dispatch_0
// CHECK-DAG:   hal.executable.binary attributes {
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = 1280071245 : i32} {
