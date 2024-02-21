// RUN: iree-opt --split-input-file --iree-flow-transformation-pipeline --iree-flow-export-benchmark-funcs --verify-diagnostics %s | FileCheck %s

// Basic usage from the `--iree-native-bindings-support` flag.

// CHECK-LABEL: func private @simpleMul
util.func public @simpleMul(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.module.export} {
  %0 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<4xf32>
  %1 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<4xf32>
  %2 = arith.mulf %0, %1 : tensor<4xf32>
  %3 = hal.tensor.export %2 : tensor<4xf32> -> !hal.buffer_view
  util.return %3 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {inlining_policy = #util.inline.never} : !hal.buffer_view
//      CHECK: util.global private @[[GLOBAL_ARG1:.+]] {inlining_policy = #util.inline.never} : !hal.buffer_view

//      CHECK: util.func public @simpleMul_benchmark() attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "entry"}} {
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
//  CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : !hal.buffer_view
// CHECK-NEXT:   %[[RET0:.+]] = util.call @simpleMul(%[[ARG0]], %[[ARG1]])
//      CHECK:   util.optimization_barrier %[[RET0]] : !hal.buffer_view
//      CHECK:   util.return

// -----

// Ensures that functions with multiple blocks are handled correctly.

util.func public @while(%start: i32, %bound: i32) -> i32 {
  cf.br ^bb1(%start : i32)
^bb1(%0: i32):
  %1 = arith.cmpi slt, %0, %bound : i32
  cf.cond_br %1, ^bb2(%0 : i32), ^bb3(%0 : i32)
^bb2(%3: i32):
  %4 = arith.addi %3, %3 : i32
  cf.br ^bb1(%4 : i32)
^bb3(%5: i32):
  util.return %5 : i32
}

//     CHECK: util.global private @[[GLOBAL_ARG0:.+]] {inlining_policy = #util.inline.never} = 0 : i32
//     CHECK: util.global private @[[GLOBAL_ARG1:.+]] {inlining_policy = #util.inline.never} = 0 : i32

//     CHECK: util.func public @while_benchmark()
// CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : i32
// CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : i32
//     CHECK:   %[[RET0:.+]] = util.call @while(%[[ARG0]], %[[ARG1]])
//     CHECK:   util.optimization_barrier %[[RET0]] : i32
//     CHECK:   util.return

// -----

// Ensure the tensors we allocate are of the desired type after casting.

// CHECK-LABEL: func private @importBufferViewBitcasting
util.func public @importBufferViewBitcasting(%view: !hal.buffer_view) -> !hal.buffer_view {
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<2xui32> as tensor<4xi32>
  %1 = arith.muli %0, %0 : tensor<4xi32>
  %2 = hal.tensor.export %1 : tensor<4xi32> -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {inlining_policy = #util.inline.never} : !hal.buffer_view
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT:.+]] = hal.tensor.export %[[SPLAT]] : tensor<4xi32> -> !hal.buffer_view
//  CHECK-DAG:   %[[DNO:.+]] = util.optimization_barrier %[[EXPORT]]
// CHECK-NEXT:   util.global.store %[[DNO]], @[[GLOBAL_ARG0]]

//      CHECK: util.func public @importBufferViewBitcasting_benchmark()
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
// CHECK-NEXT:   %[[RET0:.+]] = util.call @importBufferViewBitcasting(%[[ARG0]])
//      CHECK:   util.optimization_barrier %[[RET0]] : !hal.buffer_view
//      CHECK:   util.return

// -----

// Dynamic shape dimensions aren't supported here; we could zero them out but
// that'll likely cause confusion ((dispatches 0x0x0 work) "whoa so fast!" :).

// expected-error @+1 {{unsupported buffer view import}}
util.func public @importDynamicBufferView(%view: !hal.buffer_view) -> !hal.buffer_view {
  %dim0 = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
  %dim1 = hal.buffer_view.dim<%view : !hal.buffer_view>[1] : index
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<?x?x4xf32>{%dim0, %dim1}
  %1 = arith.mulf %0, %0 : tensor<?x?x4xf32>
  %2 = hal.tensor.export %1 : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}

// -----

// We should look for export ops to find the storage size (must be static).

// CHECK-LABEL: func private @exportBufferViewInPlace
util.func public @exportBufferViewInPlace(%view: !hal.buffer_view, %storage: !hal.buffer) -> !hal.buffer_view {
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<4xi32>
  %1 = arith.muli %0, %0 : tensor<4xi32>
  %2 = hal.tensor.export %1 into(%storage : !hal.buffer) : tensor<4xi32> -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}

//      CHECK: util.global private @[[GLOBAL_ARG0:.+]] {inlining_policy = #util.inline.never} : !hal.buffer_view
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT0:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT0:.+]] = hal.tensor.export %[[SPLAT0]] : tensor<4xi32> -> !hal.buffer_view
//  CHECK-DAG:   %[[DNO0:.+]] = util.optimization_barrier %[[EXPORT0]]
// CHECK-NEXT:   util.global.store %[[DNO0]], @[[GLOBAL_ARG0]]

//      CHECK: util.global private @[[GLOBAL_ARG1:.+]] {inlining_policy = #util.inline.never} : !hal.buffer
//      CHECK: util.initializer {
//  CHECK-DAG:   %[[SPLAT1:.+]] = flow.tensor.splat %c0_i32
//  CHECK-DAG:   %[[EXPORT1:.+]] = hal.tensor.export %[[SPLAT1]] : tensor<4xi32> -> !hal.buffer
//  CHECK-DAG:   %[[DNO1:.+]] = util.optimization_barrier %[[EXPORT1]]
// CHECK-NEXT:   util.global.store %[[DNO1]], @[[GLOBAL_ARG1]]

//      CHECK: util.func public @exportBufferViewInPlace_benchmark()
//  CHECK-DAG:   %[[ARG0:.+]] = util.global.load @[[GLOBAL_ARG0]] : !hal.buffer_view
//  CHECK-DAG:   %[[ARG1:.+]] = util.global.load @[[GLOBAL_ARG1]] : !hal.buffer
// CHECK-NEXT:   %[[RET0:.+]] = util.call @exportBufferViewInPlace(%[[ARG0]], %[[ARG1]])
//      CHECK:   util.optimization_barrier %[[RET0]] : !hal.buffer_view
//      CHECK:   util.return
