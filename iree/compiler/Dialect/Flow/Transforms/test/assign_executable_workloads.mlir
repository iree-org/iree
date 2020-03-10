// RUN: iree-opt -split-input-file -iree-flow-assign-executable-workloads %s | IreeFileCheck %s

flow.executable @singleStaticWorkload_ex_dispatch_0 {
  // CHECK-LABEL: flow.dispatch.entry @singleStaticWorkload_rgn_dispatch_0
  // CHECK-SAME: workload = dense<[4, 1, 1]> : vector<3xi32>
  flow.dispatch.entry @singleStaticWorkload_rgn_dispatch_0
  module {
    func @singleStaticWorkload_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = addf %arg0, %arg0 : tensor<4xf32>
      %1 = subf %0, %arg0 : tensor<4xf32>
      %2 = mulf %1, %arg0 : tensor<4xf32>
      return %2 : tensor<4xf32>
    }
  }
}
func @singleStaticWorkload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<[4, 1, 1]> : vector<3xi32>
  %0 = flow.dispatch @singleStaticWorkload_ex_dispatch_0::@singleStaticWorkload_rgn_dispatch_0[%cst : vector<3xi32>](%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
