// RUN: iree-opt -print-ir-after-all -split-input-file -iree-input-transformation-pipeline -iree-flow-transformation-pipeline -iree-flow-export-dispatches %s | IreeFileCheck %s

module {
  func @two_dispatch(%arg0: tensor<5x3xf32>, %arg1: tensor<3x5xf32>) -> (tensor<5x5xf32>, tensor<3x5xf32>) attributes { iree.module.export } {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
    %1 = "mhlo.dot"(%arg1, %0) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
    return %0, %1 : tensor<5x5xf32>, tensor<3x5xf32>
  }
}
// CHECK-DAG: flow.variable @[[IN0_0:.+]] dense<{{.*}}> : tensor<5x3xf32>
// CHECK-DAG: flow.variable @[[IN0_1:.+]] dense<{{.*}}> : tensor<3x5xf32>
//     CHECK: func @two_dispatch_ex_dispatch_0_entry
//     CHECK: %{{.+}} = flow.variable.load @[[IN0_0]] : tensor<5x3xf32>
//     CHECK: %{{.+}} = flow.variable.load @[[IN0_1]] : tensor<3x5xf32>
//     CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<5x5xf32> =
//     CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_0::@two_dispatch_ex_dispatch_0[%{{.+}}](%{{.+}}, %{{.+}}) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
//     CHECK:   flow.return %[[DISPATCH_RES]] : tensor<5x5xf32>
//     CHECK: iree.do_not_optimize(%[[RES]]) : tensor<5x5xf32>
//
// CHECK-DAG: flow.variable @[[IN1_0:.+]] dense<{{.*}}> : tensor<3x5xf32>
// CHECK-DAG: flow.variable @[[IN1_1:.+]] dense<{{.*}}> : tensor<5x5xf32>
//     CHECK: func @two_dispatch_ex_dispatch_1_entry
//     CHECK: %{{.+}} = flow.variable.load @[[IN1_0]] : tensor<3x5xf32>
//     CHECK: %{{.+}} = flow.variable.load @[[IN1_1]] : tensor<5x5xf32>
//     CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<3x5xf32>
//     CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_1::@two_dispatch_ex_dispatch_1[%{{.+}}](%{{.+}}, %{{.+}}) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
//     CHECK:   flow.return %[[DISPATCH_RES]] : tensor<3x5xf32>
//     CHECK: iree.do_not_optimize(%[[RES]]) : tensor<3x5xf32>
//
// CHECK-DAG: flow.variable @[[MAIN_IN_0:.+]] dense<{{.*}}> : tensor<5x3xf32>
// CHECK-DAG: flow.variable @[[MAIN_IN_1:.+]] dense<{{.*}}> : tensor<3x5xf32>
//     CHECK: func @two_dispatch_dummy_args()
//     CHECK: %{{.+}} = flow.variable.load @[[MAIN_IN_0]] : tensor<5x3xf32>
//     CHECK: %{{.+}} = flow.variable.load @[[MAIN_IN_1]] : tensor<3x5xf32>
//     CHECK: flow.ex.stream.fragment({{.+}}) -> (tensor<5x5xf32>, tensor<3x5xf32>) =
//     CHECK:   %[[DISPATCH_RES1:.+]] = flow.dispatch
//     CHECK:   %[[DISPATCH_RES2:.+]] = flow.dispatch
//     CHECK:   flow.return %[[DISPATCH_RES1]], %[[DISPATCH_RES2]] : tensor<5x5xf32>, tensor<3x5xf32>

// -----

func @while(%start: tensor<i32>, %bound: tensor<i32>) -> tensor<i32> attributes { iree.module.export }  {
  %res = "mhlo.while"(%start) ( {
  ^bb0(%count: tensor<i32>):
    %1 = "mhlo.compare"(%count, %bound) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%count: tensor<i32>):
    %1 = mhlo.add %count, %count : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  return %res : tensor<i32>
}

// CHECK: func @while_dummy_args
// CHECK: %{{.+}} = flow.variable.load @{{.+}} : tensor<i32>
// CHECK: %{{.+}} = flow.variable.load @{{.+}} : tensor<i32>
// CHECK: br ^bb1
