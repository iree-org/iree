// RUN: iree-opt -iree-flow-transformation-pipeline -iree-flow-export-dispatches %s | IreeFileCheck %s

module {
  func @two_dispatch() -> (tensor<5x5xf32>, tensor<3x5xf32>) attributes { iree.module.export } {
    %0 = iree.unfoldable_constant dense<1.0> : tensor<5x3xf32>
    %1 = iree.unfoldable_constant dense<0.4> : tensor<3x5xf32>
    %2 = "mhlo.dot"(%0, %1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
    %3 = "mhlo.dot"(%1, %2) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
    return %2, %3 : tensor<5x5xf32>, tensor<3x5xf32>
  }
}
// CHECK: func @two_dispatch_ex_dispatch_0_entry
// CHECK: %[[CST0:.+]] = constant dense<0.000000e+00> : tensor<5x3xf32>
// CHECK: %[[CST1:.+]] = constant dense<0.000000e+00> : tensor<3x5xf32>
// CHECK: %{{.+}} = iree.do_not_optimize(%[[CST0]]) : tensor<5x3xf32>
// CHECK: %{{.+}} = iree.do_not_optimize(%[[CST1]]) : tensor<3x5xf32>
// CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<5x5xf32> {
// CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_0::@two_dispatch_ex_dispatch_0[%{{.+}} : index](%{{.+}}, %{{.+}}) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
// CHECK:   flow.return %[[DISPATCH_RES]] : tensor<5x5xf32>
// CHECK: return %[[RES]] : tensor<5x5xf32>

// CHECK: func @two_dispatch_ex_dispatch_1_entry
// CHECK: %[[CST0:.+]] = constant dense<0.000000e+00> : tensor<3x5xf32>
// CHECK: %[[CST1:.+]] = constant dense<0.000000e+00> : tensor<5x5xf32>
// CHECK: %[[ARG0:.+]] = iree.do_not_optimize(%[[CST0]]) : tensor<3x5xf32>
// CHECK: %[[ARG1:.+]] = iree.do_not_optimize(%[[CST1]]) : tensor<5x5xf32>
// CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<3x5xf32>
// CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_1::@two_dispatch_ex_dispatch_1[%{{.+}} : index](%{{.+}}, %{{.+}}) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK:   flow.return %[[DISPATCH_RES]] : tensor<3x5xf32>
// CHECK: return %[[RES]] : tensor<3x5xf32>
