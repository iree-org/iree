// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=sync})' --split-input-file %s | FileCheck %s

// Tests basic dynamic tensor I/O marshaling.

// CHECK-LABEL: util.func public @dynamicEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME:   iree.reflection =
//  CHECK-SAME:       iree.abi.declaration = "sync func @dynamicEntry(%input0: tensor<?x8x8x3xf32>, %input1: tensor<?x8x8x3xf32>) -> (%output0: tensor<?x8x8x3xf32>, %output1: tensor<?x8x8x3xf32>)"
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_DIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import %[[ARG0]] "input0" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG0_DIM0]]}
//  CHECK-NEXT:   %[[ARG1_DIM0:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.import %[[ARG1]] "input1" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG1_DIM0]]}
//  CHECK-NEXT:   %[[RET_TENSORS:.+]]:2 = util.call @_dynamicEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//       CHECK:   %[[RET0_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#0, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#0 "output0" : tensor<?x8x8x3xf32>{%[[RET0_DIM0]]} -> !hal.buffer_view
//       CHECK:   %[[RET1_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#1, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#1 "output1" : tensor<?x8x8x3xf32>{%[[RET1_DIM0]]} -> !hal.buffer_view
//  CHECK-NEXT:   util.return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: util.func private @_dynamicEntry(
util.func public @dynamicEntry(%arg0: tensor<?x8x8x3xf32>, %arg1: tensor<?x8x8x3xf32>) ->
    (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<?x8x8x3xf32>
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  util.return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// Tests that an existing iree.reflection dictionary is merged with the new
// reflection information.

// CHECK-LABEL: util.func public @existingReflection
// CHECK-SAME: iree.reflection =
// CHECK-SAME:     iree.abi.declaration = "sync func @existingReflection
// CHECK-SAME:     some.attr = 4 : index
// CHECK: util.func private @_existingReflection
// CHECK-NOT: iree.reflection = {some.attr = 4 : index}
util.func public @existingReflection() attributes {
  iree.reflection = {
    some.attr = 4 : index
  }
} {
  util.return
}

// -----

// Tests that iree.abi.declaration is added when needed and otherwise the user
// provided value is passed through.

// CHECK-LABEL: util.func public @existingDeclaration
// CHECK-SAME: iree.reflection =
// CHECK-SAME:     iree.abi.declaration = "some.python.thing(types_are_overrated)"
util.func public @existingDeclaration(%arg0: tensor<i32>) attributes {
  iree.abi.declaration = "some.python.thing(types_are_overrated)"
} {
  util.return
}

// -----

// Tests that name overrides propagate into both metadata and assertion IR.

// CHECK-LABEL: util.func public @namedEntry
// CHECK-SAME: iree.reflection =
// CHECK-SAME:     iree.abi.declaration = "sync func @namedEntry(%my_input_0: tensor<3xf32>, %input1: tensor<3xf32>) -> (%my_output_0: tensor<3xf32>, %output1: tensor<3xf32>)"
util.func public @namedEntry(%arg0: tensor<3xf32> {iree.abi.name = "my_input_0"}, %arg1: tensor<3xf32>) ->
    (tensor<3xf32> {iree.abi.name = "my_output_0"}, tensor<3xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<3xf32>
  util.return %0, %0 : tensor<3xf32>, tensor<3xf32>
}

// -----

// Tests that exports with encodings specified are propagated to the HAL ops.

// CHECK-LABEL: util.func public @exportEncodings
//  CHECK-SAME:   iree.abi.declaration = "sync func @exportEncodings(%input0: tensor<?x8x8x3xf32> {iree.abi.encoding = tensor<?x8x8x3xi32>}) -> (%output0: tensor<?x8x8x3xf32> {iree.abi.encoding = tensor<?x8x8x3xi32>})"
// CHECK: hal.tensor.import {{.+}} : !hal.buffer_view -> tensor<?x8x8x3xi32> as tensor<?x8x8x3xf32>{{.+}}
// CHECK: hal.tensor.export {{.+}} : tensor<?x8x8x3xi32> as tensor<?x8x8x3xf32>{{.+}} -> !hal.buffer_view

// CHECK-LABEL: util.func private @_exportEncodings
util.func public @exportEncodings(%arg0: tensor<?x8x8x3xf32> {iree.abi.encoding = tensor<?x8x8x3xi32>}) -> (tensor<?x8x8x3xf32> {iree.abi.encoding = tensor<?x8x8x3xi32>}) {
  util.return %arg0 : tensor<?x8x8x3xf32>
}

// -----

// Tests specifying explicit storage for specific function results.

// CHECK-LABEL: util.func public @outputStorage
//  CHECK-SAME:   (%[[ARG0:[a-z0-9]+]]: !hal.buffer_view, %[[RET1_STORAGE:[a-z0-9]+]]: !hal.buffer)
//  CHECK-SAME: -> (!hal.buffer_view, !hal.buffer_view) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME:   iree.reflection =
//  CHECK-SAME:       iree.abi.declaration = "sync func @outputStorage(%input0: tensor<?x8x8x3xf32>, %input1: !hal.buffer {iree.abi.output = 1 : index}) -> (%output0: tensor<?x8x8x3xf32>, %output1: tensor<?x8x8x3xf32>)"
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_DIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import %[[ARG0]] "input0" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG0_DIM0]]}
//  CHECK-NEXT:   %[[RET_TENSORS:.+]]:2 = util.call @_outputStorage(%[[ARG0_TENSOR]], %[[RET1_STORAGE]])
//       CHECK:   %[[RET0_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#0, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#0 "output0" : tensor<?x8x8x3xf32>{%[[RET0_DIM0]]} -> !hal.buffer_view
//       CHECK:   %[[RET1_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#1, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#1 "output1" into(%[[RET1_STORAGE]] : !hal.buffer) : tensor<?x8x8x3xf32>{%[[RET1_DIM0]]} -> !hal.buffer_view
//  CHECK-NEXT:   util.return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: util.func private @_outputStorage(
util.func public @outputStorage(%arg0: tensor<?x8x8x3xf32>, %ret1: !hal.buffer {iree.abi.output = 1 : index}) ->
    (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) {
  %0 = arith.addf %arg0, %arg0 : tensor<?x8x8x3xf32>
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  util.return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// Tests that functions already wrapped (iree.abi.stub present) are ignored.

// CHECK-LABEL: util.func public @wrappedAlready
//  CHECK-SAME: (%arg0: !hal.buffer_view) -> !hal.buffer_view
//  CHECK-SAME: attributes {iree.abi.stub}
util.func public @wrappedAlready(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  util.return %arg0 : !hal.buffer_view
}
// CHECK-NOT: util.func public @_wrappedAlready

// -----

// Tests that a function calling an exported function is redirected to the
// original unwrapped call.

// CHECK-LABEL: util.func public @exportA(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK:   call @_exportA
// CHECK: util.func private @_exportA(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:   util.return %arg0
util.func public @exportA(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  util.return %arg0 : tensor<?x?xi32>
}

// CHECK: util.func public @exportB(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK:   util.call @_exportB
// CHECK: util.func private @_exportB(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:   util.call @_exportA
util.func public @exportB(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = util.call @exportA(%arg0) : (tensor<?x?xi32>) -> tensor<?x?xi32>
  util.return %0 : tensor<?x?xi32>
}

// -----

// Tests that imported functions get converted to canonical ABI types and
// wrapper functions are built to preserve internal behavior.

// CHECK-LABEL: util.func private @import(%arg0: !hal.buffer_view) -> !hal.buffer_view
util.func private @import(tensor<?x2xi32>) -> tensor<2x?xi32>

// CHECK: util.func private @_import(%[[ARG_TENSOR:.+]]: tensor<?x2xi32>) -> tensor<2x?xi32> {
// CHECK:   %[[ARG_DIM:.+]] = tensor.dim %[[ARG_TENSOR]], %c0
// CHECK:   %[[ARG_VIEW:.+]] = hal.tensor.export %[[ARG_TENSOR]] : tensor<?x2xi32>{%[[ARG_DIM]]} -> !hal.buffer_view
// CHECK:   %[[RET_VIEW:.+]] = util.call @import(%[[ARG_VIEW]]) : (!hal.buffer_view) -> !hal.buffer_view
// CHECK:   %[[RET_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEW]] : !hal.buffer_view>[1]
// CHECK:   %[[RET_TENSOR:.+]] = hal.tensor.import %[[RET_VIEW]] : !hal.buffer_view -> tensor<2x?xi32>{%[[RET_DIM]]}
// CHECK:   util.return %[[RET_TENSOR]]
// CHECK: }

// CHECK: util.func private @caller(%arg0: tensor
util.func private @caller(%arg0: tensor<?x2xi32>) -> tensor<2x?xi32> {
  // CHECK: util.call @_import(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  %0 = util.call @import(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  util.return %0 : tensor<2x?xi32>
}

// -----

// Tests that imports with encodings specified are propagated to the HAL ops.

// CHECK-LABEL: util.func private @importEncodings(%arg0: !hal.buffer_view) -> !hal.buffer_view
util.func private @importEncodings(tensor<?x2xi32> {iree.abi.encoding = tensor<?x2xf32>}) -> (tensor<2x?xi32> {iree.abi.encoding = tensor<2x?xf32>})

// CHECK: util.func private @_importEncodings(%[[ARG_TENSOR:.+]]: tensor<?x2xi32>) -> tensor<2x?xi32> {
// CHECK:   %[[ARG_DIM:.+]] = tensor.dim %[[ARG_TENSOR]], %c0
// CHECK:   %[[ARG_VIEW:.+]] = hal.tensor.export %[[ARG_TENSOR]] : tensor<?x2xi32>{%[[ARG_DIM]]} -> !hal.buffer_view
// CHECK:   %[[RET_VIEW:.+]] = util.call @importEncodings(%[[ARG_VIEW]]) : (!hal.buffer_view) -> !hal.buffer_view
// CHECK:   %[[RET_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEW]] : !hal.buffer_view>[1]
// CHECK:   %[[RET_TENSOR:.+]] = hal.tensor.import %[[RET_VIEW]] : !hal.buffer_view -> tensor<2x?xi32>{%[[RET_DIM]]}
// CHECK:   util.return %[[RET_TENSOR]]
// CHECK: }

// CHECK: util.func private @importEncodingsCaller(%arg0: tensor
util.func private @importEncodingsCaller(%arg0: tensor<?x2xi32>) -> tensor<2x?xi32> {
  // CHECK: call @_importEncodings(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  %0 = util.call @importEncodings(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  util.return %0 : tensor<2x?xi32>
}
