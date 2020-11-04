// RUN: iree-opt -split-input-file -iree-flow-dedupliclate-executables %s | IreeFileCheck %s

// CHECK-LABEL: flow.executable @single_executable_ex_dispatch_0
flow.executable @single_executable_ex_dispatch_0 {
  flow.dispatch.entry @single_executable_rgn_dispatch_0
  module {
    func @single_executable_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @single_executable
func @single_executable(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @single_executable_ex_dispatch_0::@single_executable_rgn_dispatch_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @single_executable_ex_dispatch_0::@single_executable_rgn_dispatch_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: flow.executable @duplicate_executables_ex_dispatch_0
flow.executable @duplicate_executables_ex_dispatch_0 {
  flow.dispatch.entry @duplicate_executables_rgn_dispatch_0
  module {
    func @duplicate_executables_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-NOT: flow.executable @duplicate_executables_ex_dispatch_1
flow.executable @duplicate_executables_ex_dispatch_1 {
  flow.dispatch.entry @duplicate_executables_rgn_dispatch_1
  module {
    func @duplicate_executables_rgn_dispatch_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: flow.executable @duplicate_executables_ex_dispatch_2
flow.executable @duplicate_executables_ex_dispatch_2 {
  flow.dispatch.entry @duplicate_executables_rgn_dispatch_2
  module {
    func @duplicate_executables_rgn_dispatch_2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.subtract %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}
// CHECK-LABEL: func @duplicate_executables
func @duplicate_executables(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = constant 4 : index
  // CHECK: %0 = flow.dispatch @duplicate_executables_ex_dispatch_0::@duplicate_executables_rgn_dispatch_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @duplicate_executables_ex_dispatch_0::@duplicate_executables_rgn_dispatch_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %1 = flow.dispatch @duplicate_executables_ex_dispatch_0::@duplicate_executables_rgn_dispatch_0[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @duplicate_executables_ex_dispatch_1::@duplicate_executables_rgn_dispatch_1[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: %2 = flow.dispatch @duplicate_executables_ex_dispatch_2::@duplicate_executables_rgn_dispatch_2[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = flow.dispatch @duplicate_executables_ex_dispatch_2::@duplicate_executables_rgn_dispatch_2[%c4 : index](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}


// TODO(scotttodd): example with multiple flow.dispatch.entry ops
