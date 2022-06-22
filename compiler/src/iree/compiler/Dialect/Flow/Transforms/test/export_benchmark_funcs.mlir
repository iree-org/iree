// RUN: iree-opt --split-input-file --iree-mhlo-input-transformation-pipeline --iree-flow-transformation-pipeline --iree-flow-export-benchmark-funcs --verify-diagnostics %s | FileCheck %s

module {
  func.func @two_dispatch(%arg0: tensor<5x3xf32>, %arg1: tensor<3x5xf32>) -> (tensor<5x5xf32>, tensor<3x5xf32>) {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
    %1 = "mhlo.dot"(%arg1, %0) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
    return %0, %1 : tensor<5x5xf32>, tensor<3x5xf32>
  }
}

//  CHECK-DAG: util.global private @[[GLOBAL_ARG0:.+]] {noinline} = dense<{{.*}}> : tensor<5x3xf32>
//  CHECK-DAG: util.global private @[[GLOBAL_ARG1:.+]] {noinline} = dense<{{.*}}> : tensor<3x5xf32>

//      CHECK: func.func @two_dispatch_benchmark()
// CHECK-SAME: attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "entry"}}
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : tensor<5x3xf32>
//  CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : tensor<3x5xf32>
//      CHECK:   %[[RET:.+]]:2 = call @two_dispatch(%[[ARG0]], %[[ARG1]])
//  CHECK-DAG:   util.do_not_optimize(%[[RET]]#0) : tensor<5x5xf32>
//  CHECK-DAG:   util.do_not_optimize(%[[RET]]#1) : tensor<3x5xf32>

// -----

func.func @while(%start: tensor<i32>, %bound: tensor<i32>) -> tensor<i32> {
  cf.br ^bb1(%start : tensor<i32>)
^bb1(%0: tensor<i32>):
  %1 = "mhlo.compare"(%0, %bound) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = tensor.extract %1[] : tensor<i1>
  cf.cond_br %2, ^bb2(%0 : tensor<i32>), ^bb3(%0 : tensor<i32>)
^bb2(%3: tensor<i32>):
  %4 = mhlo.add %3, %3 : tensor<i32>
  cf.br ^bb1(%4 : tensor<i32>)
^bb3(%5: tensor<i32>):
  return %5 : tensor<i32>
}

//     CHECK: util.global private @[[GLOBAL_ARG0:.+]] {noinline} = dense<0> : tensor<i32>
//     CHECK: util.global private @[[GLOBAL_ARG1:.+]] {noinline} = dense<0> : tensor<i32>

//     CHECK: func.func @while_benchmark()
// CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : tensor<i32>
// CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : tensor<i32>
//     CHECK:   %[[RET0:.+]] = call @while(%[[ARG0]], %[[ARG1]])
//     CHECK:   util.do_not_optimize(%[[RET0]]) : tensor<i32>
//     CHECK:   return

// -----

// Basic usage from the `--iree-native-bindings-support` flag.

// CHECK-LABEL: func private @simpleMul
func.func @simpleMul(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export} {
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<4xf32>
  %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<4xf32>
  %2 = mhlo.multiply %0, %1 {name = "mul.1"} : tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  return %3 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {noinline} : !hal.buffer_view
//      CHECK: util.global private @[[GLOBAL_ARG1:.+]] {noinline} : !hal.buffer_view

//      CHECK: func.func @simpleMul_benchmark() attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "entry"}} {
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
//  CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : !hal.buffer_view
// CHECK-NEXT:   %[[RET0:.+]] = call @simpleMul(%[[ARG0]], %[[ARG1]])
//      CHECK:   util.do_not_optimize(%[[RET0]]) : !hal.buffer_view
//      CHECK:   return

// -----

// Ensure the tensors we allocate are of the desired type after casting.

// CHECK-LABEL: func private @importBufferViewBitcasting
func.func @importBufferViewBitcasting(%view: !hal.buffer_view) -> !hal.buffer_view {
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<2xui32> as tensor<4xi32>
  %1 = mhlo.multiply %0, %0 {name = "mul.3"} : tensor<4xi32>
  %2 = hal.tensor.export %1 : tensor<4xi32> -> !hal.buffer_view
  return %2 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {noinline} : !hal.buffer_view
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT:.+]] = hal.tensor.export %[[SPLAT]] : tensor<4xi32> -> !hal.buffer_view
//  CHECK-DAG:   %[[DNO:.+]] = util.do_not_optimize(%[[EXPORT]])
// CHECK-NEXT:   util.global.store %[[DNO]], @[[GLOBAL_ARG0]]

//      CHECK: func.func @importBufferViewBitcasting_benchmark()
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
// CHECK-NEXT:   %[[RET0:.+]] = call @importBufferViewBitcasting(%[[ARG0]])
//      CHECK:   util.do_not_optimize(%[[RET0]]) : !hal.buffer_view
//      CHECK:   return

// -----

// Dynamic shape dimensions aren't supported here; we could zero them out but
// that'll likely cause confusion ((dispatches 0x0x0 work) "whoa so fast!" :).

// expected-error @+1 {{unsupported buffer view import}}
func.func @importDynamicBufferView(%view: !hal.buffer_view) -> !hal.buffer_view {
  %dim0 = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
  %dim1 = hal.buffer_view.dim<%view : !hal.buffer_view>[1] : index
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<?x?x4xf32>{%dim0, %dim1}
  %1 = mhlo.multiply %0, %0 {name = "mul.2"} : tensor<?x?x4xf32>
  %2 = hal.tensor.export %1 : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  return %2 : !hal.buffer_view
}

// -----

// We should look for export ops to find the storage size (must be static).

// CHECK-LABEL: func private @exportBufferViewInPlace
func.func @exportBufferViewInPlace(%view: !hal.buffer_view, %storage: !hal.buffer) -> !hal.buffer_view {
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<4xi32>
  %1 = mhlo.multiply %0, %0 {name = "mul.4"} : tensor<4xi32>
  %2 = hal.tensor.export %1 into %storage : tensor<4xi32> -> !hal.buffer_view
  return %2 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {noinline} : !hal.buffer_view
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT0:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT0:.+]] = hal.tensor.export %[[SPLAT0]] : tensor<4xi32> -> !hal.buffer_view
//  CHECK-DAG:   %[[DNO0:.+]] = util.do_not_optimize(%[[EXPORT0]])
// CHECK-NEXT:   util.global.store %[[DNO0]], @[[GLOBAL_ARG0]]

//      CHECK: util.global private @[[GLOBAL_ARG1:.+]] {noinline} : !hal.buffer
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT1:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT1:.+]] = hal.tensor.export %[[SPLAT1]] : tensor<4xi32> -> !hal.buffer
//  CHECK-DAG:   %[[DNO1:.+]] = util.do_not_optimize(%[[EXPORT1]])
// CHECK-NEXT:   util.global.store %[[DNO1]], @[[GLOBAL_ARG1]]

//      CHECK: func.func @exportBufferViewInPlace_benchmark()
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
//  CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : !hal.buffer
// CHECK-NEXT:   %[[RET0:.+]] = call @exportBufferViewInPlace(%[[ARG0]], %[[ARG1]])
//      CHECK:   util.do_not_optimize(%[[RET0]]) : !hal.buffer_view
//      CHECK:   return
