// RUN: iree-opt -split-input-file -iree-input-transformation-pipeline -iree-flow-transformation-pipeline -iree-flow-export-benchmark-funcs %s | IreeFileCheck %s

module {
  func @two_dispatch(%arg0: tensor<5x3xf32>, %arg1: tensor<3x5xf32>) -> (tensor<5x5xf32>, tensor<3x5xf32>) attributes { iree.module.export } {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
    %1 = "mhlo.dot"(%arg1, %0) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
    return %0, %1 : tensor<5x5xf32>, tensor<3x5xf32>
  }
}
// CHECK-DAG: flow.variable @[[IN0_0:.+]] dense<{{.*}}> : tensor<5x3xf32>
// CHECK-DAG: flow.variable @[[IN0_1:.+]] dense<{{.*}}> : tensor<3x5xf32>
//     CHECK: func @two_dispatch_ex_dispatch_0_benchmark
// CHECK-DAG: %{{.+}} = flow.variable.load @[[IN0_0]] : tensor<5x3xf32>
// CHECK-DAG: %{{.+}} = flow.variable.load @[[IN0_1]] : tensor<3x5xf32>
//     CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<5x5xf32> =
//     CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_0::@two_dispatch_ex_dispatch_0[%{{.+}}](%{{.+}}, %{{.+}}) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
//     CHECK:   flow.return %[[DISPATCH_RES]] : tensor<5x5xf32>
//     CHECK: iree.do_not_optimize(%[[RES]]) : tensor<5x5xf32>

// CHECK-DAG: flow.variable @[[IN1_0:.+]] dense<{{.*}}> : tensor<3x5xf32>
// CHECK-DAG: flow.variable @[[IN1_1:.+]] dense<{{.*}}> : tensor<5x5xf32>
//     CHECK: func @two_dispatch_ex_dispatch_1_benchmark
// CHECK-DAG: %{{.+}} = flow.variable.load @[[IN1_0]] : tensor<3x5xf32>
// CHECK-DAG: %{{.+}} = flow.variable.load @[[IN1_1]] : tensor<5x5xf32>
//     CHECK: %[[RES:.+]] = flow.ex.stream.fragment({{.+}}) -> tensor<3x5xf32>
//     CHECK:   %[[DISPATCH_RES:.+]] = flow.dispatch @two_dispatch_ex_dispatch_1::@two_dispatch_ex_dispatch_1[%{{.+}}](%{{.+}}, %{{.+}}) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
//     CHECK:   flow.return %[[DISPATCH_RES]] : tensor<3x5xf32>
//     CHECK: iree.do_not_optimize(%[[RES]]) : tensor<3x5xf32>

// CHECK-DAG: flow.variable @[[MAIN_IN_0:.+]] dense<{{.*}}> : tensor<5x3xf32>
// CHECK-DAG: flow.variable @[[MAIN_IN_1:.+]] dense<{{.*}}> : tensor<3x5xf32>
//     CHECK: func @two_dispatch_benchmark()
// CHECK-DAG: %[[ARG0:.+]] = flow.variable.load @[[MAIN_IN_0]] : tensor<5x3xf32>
// CHECK-DAG: %[[ARG1:.+]] = flow.variable.load @[[MAIN_IN_1]] : tensor<3x5xf32>
//     CHECK: %[[RET:.+]]:2 = call @two_dispatch(%[[ARG0]], %[[ARG1]])
// CHECK-DAG: iree.do_not_optimize(%[[RET]]#0) : tensor<5x5xf32>
// CHECK-DAG: iree.do_not_optimize(%[[RET]]#1) : tensor<3x5xf32>

// -----

func @while(%start: tensor<i32>, %bound: tensor<i32>) -> tensor<i32> attributes {iree.module.export} {
  br ^bb1(%start : tensor<i32>)
^bb1(%0: tensor<i32>):
  %1 = "mhlo.compare"(%0, %bound) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = tensor.extract %1[] : tensor<i1>
  cond_br %2, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)
^bb2(%3: tensor<i32>):
  %4 = mhlo.add %3, %3 : tensor<i32>
  br ^bb1(%4 : tensor<i32>)
^bb3(%5: tensor<i32>):
  return %5 : tensor<i32>
}

//     CHECK: flow.variable @_benchmark_input_0 dense<0> : tensor<i32> attributes {noinline, sym_visibility = "private"}
//     CHECK: flow.variable @_benchmark_input_1 dense<0> : tensor<i32> attributes {noinline, sym_visibility = "private"}
//     CHECK: func @while_benchmark() attributes {iree.abi.stub, iree.module.export, iree.reflection = {benchmark = "entry"}} {
// CHECK-DAG:   %[[ARG0:.+]] = flow.variable.load @_benchmark_input_0 : tensor<i32>
// CHECK-DAG:   %[[ARG1:.+]] = flow.variable.load @_benchmark_input_1 : tensor<i32>
//     CHECK:   %[[RET0:.+]] = call @while(%[[ARG0]], %[[ARG1]])
//     CHECK:   iree.do_not_optimize(%[[RET0]]) : tensor<i32>
//     CHECK:   return
//     CHECK: }
