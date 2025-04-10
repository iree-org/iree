// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @tensorReshape
util.func public @tensorReshape(%arg0 : tensor<4x4xf32>) -> tensor<16xf32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xf32> -> tensor<16xf32>
  util.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeScalar
util.func public @tensorReshapeScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  %0 = flow.tensor.reshape %arg0 : tensor<f32> -> tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorReshapeDynamic
util.func public @tensorReshapeDynamic(%arg0 : tensor<?x4xf32>) -> tensor<?x2xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // CHECK: %0 = flow.tensor.reshape %arg0 : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c8}
  %0 = flow.tensor.reshape %arg0 : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c8}
  util.return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeComplex
util.func public @tensorReshapeComplex(%arg0 : tensor<4x4xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  // CHECK-NEXT: flow.tensor.reshape %arg0 : tensor<4x4xcomplex<f32>> -> tensor<16xcomplex<f32>>
  %0 = flow.tensor.reshape %arg0 : tensor<4x4xcomplex<f32>> -> tensor<16xcomplex<f32>>
  util.return %0 : tensor<16xcomplex<f32>>
}

// -----

// CHECK-LABEL: @tensorBitCast
util.func public @tensorBitCast(%arg0 : tensor<16xi32>) -> tensor<4x8xi16> {
  // CHECK-NEXT: %0 = flow.tensor.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  %0 = flow.tensor.bitcast %arg0 : tensor<16xi32> -> tensor<4x8xi16>
  util.return %0 : tensor<4x8xi16>
}

// -----

// CHECK-LABEL: @tensorLoad
util.func public @tensorLoad(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<4x4xf32>
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @tensorLoadScalar
util.func public @tensorLoadScalar(%arg0 : tensor<f32>) -> f32 {
  // CHECK-NEXT: %0 = flow.tensor.load %arg0 : tensor<f32>
  %0 = flow.tensor.load %arg0 : tensor<f32>
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @tensorLoadDynamic
util.func public @tensorLoadDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.load %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  util.return %0 : f32
}

// -----

// CHECK-LABEL: @tensorStore
util.func public @tensorStore(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index, %arg3 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @tensorStoreScalar
util.func public @tensorStoreScalar(%arg0 : f32, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  %0 = flow.tensor.store %arg0, %arg1 : tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorStoreDynamic
util.func public @tensorStoreDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index, %arg3 : f32) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.store %arg3, %arg0[%arg1, %arg2] : tensor<?x4xf32>{%c4}
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorAlloca
util.func public @tensorAlloca(%arg0: index) -> tensor<?x0x1xf32> {
  // CHECK-NEXT: = flow.tensor.alloca : tensor<?x0x1xf32>{%arg0}
  %0 = flow.tensor.alloca : tensor<?x0x1xf32>{%arg0}
  util.return %0 : tensor<?x0x1xf32>
}

// -----

// CHECK-LABEL: @tensorEmpty
util.func public @tensorEmpty(%arg0: index) -> tensor<?x0x1xf32> {
  // CHECK-NEXT: = flow.tensor.empty : tensor<?x0x1xf32>{%arg0}
  %0 = flow.tensor.empty : tensor<?x0x1xf32>{%arg0}
  util.return %0 : tensor<?x0x1xf32>
}

// -----

// CHECK-LABEL: @tensorSplat
util.func public @tensorSplat(%arg0 : f32) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.splat %arg0 : tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @tensorSplatScalar
util.func public @tensorSplatScalar(%arg0 : f32) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.splat %arg0 : tensor<f32>
  %0 = flow.tensor.splat %arg0 : tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorSplatDynamic
util.func public @tensorSplatDynamic(%arg0 : f32) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.splat %arg0 : tensor<?x4xf32>{%c4}
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorClone
util.func public @tensorClone(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  %0 = flow.tensor.clone %arg0 : tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @tensorTransfer
util.func public @tensorTransfer(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.transfer %arg0 : tensor<4x4xf32> to "dummy"
  %0 = flow.tensor.transfer %arg0 : tensor<4x4xf32> to "dummy"
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @tensorCloneScalar
util.func public @tensorCloneScalar(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = flow.tensor.clone %arg0 : tensor<f32>
  %0 = flow.tensor.clone %arg0 : tensor<f32>
  util.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensorCloneDynamic
util.func public @tensorCloneDynamic(%arg0 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.clone %arg0 : tensor<?x4xf32>{%c4}
  %0 = flow.tensor.clone %arg0 : tensor<?x4xf32>{%c4}
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @tensorEncodeStatic
// CHECK-SAME:    %[[ARG0:.[a-zA-Z0-9]+]]
#encoding = #iree_encoding.testing_encoding<>
util.func public @tensorEncodeStatic(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32, #encoding> {
  // CHECK: %[[RES:.+]] = flow.tensor.encode %[[ARG0]] : tensor<4x4xf32> -> tensor<4x4xf32, #[[$ENCODING]]>
  %0 = flow.tensor.encode %arg0 : tensor<4x4xf32> -> tensor<4x4xf32, #encoding>
  util.return %0 : tensor<4x4xf32, #encoding>
}

// -----

// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-LABEL: @tensorEncodeDynamic
// CHECK-SAME:    %[[ARG0:.[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:.[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:.[a-zA-Z0-9]+]]
#encoding = #iree_encoding.testing_encoding<>
util.func public @tensorEncodeDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2: index) -> tensor<5x?xf32, #encoding> {
  // CHECK: %[[RES:.+]] = flow.tensor.encode %[[ARG0]] : tensor<?x4xf32>{%[[ARG1]]} -> tensor<5x?xf32, #[[$ENCODING]]>{%[[ARG2]]}
  %0 = flow.tensor.encode %arg0 : tensor<?x4xf32>{%arg1} -> tensor<5x?xf32, #encoding>{%arg2}
  util.return %0 : tensor<5x?xf32, #encoding>
}

// -----

// CHECK-LABEL: @tensorEncodeChangeEncoding
// CHECK-SAME:    %[[ARG0:.[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:.[a-zA-Z0-9]+]]
#encoding = #iree_encoding.unspecialized_encoding<123>
#encoding1 = #iree_encoding.unspecialized_encoding<456>
util.func public @tensorEncodeChangeEncoding(%arg0 : tensor<?x4xf32, #encoding>, %arg1 : index) -> tensor<?x4xf32, #encoding1> {
  // CHECK:      %[[RES:.+]] = flow.tensor.encode %[[ARG0]]
  // CHECK-SAME:   : tensor<?x4xf32, #iree_encoding.unspecialized_encoding<123>>{%[[ARG1]]}
  // CHECK-SAME:   -> tensor<?x4xf32, #iree_encoding.unspecialized_encoding<456>>{%[[ARG1]]}
  %0 = flow.tensor.encode %arg0 : tensor<?x4xf32, #encoding>{%arg1} -> tensor<?x4xf32, #encoding1>{%arg1}
  util.return %0 : tensor<?x4xf32, #encoding1>
}

// -----

// CHECK-LABEL: @tensorSlice
util.func public @tensorSlice(%arg0 : tensor<4x4xf32>, %arg1 : index, %arg2 : index) -> tensor<2x2xf32> {
  // CHECK-NEXT: %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<4x4xf32> -> tensor<2x2xf32>
  util.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: @tensorSliceDynamic
util.func public @tensorSliceDynamic(%arg0 : tensor<?x4xf32>, %arg1 : index, %arg2 : index) -> tensor<?x2xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // CHECK: %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c2}
  %0 = flow.tensor.slice %arg0[%arg1, %arg2 for %arg2, %arg1] : tensor<?x4xf32>{%c4} -> tensor<?x2xf32>{%c2}
  util.return %0 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
util.func public @tensorUpdate(%arg0 : tensor<2x2xf32>, %arg1 : tensor<4x4xf32>, %arg2 : index, %arg3 : index) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> %arg1 as tensor<4x4xf32>
  %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<2x2xf32> -> %arg1 as tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @tensorUpdateDynamic
util.func public @tensorUpdateDynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x4xf32>, %arg2 : index, %arg3 : index) -> tensor<?x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<?x?xf32>{%c1, %c2} -> %arg1 as tensor<?x4xf32>{%c3}
  %0 = flow.tensor.update %arg0, %arg1[%arg2, %arg3] : tensor<?x?xf32>{%c1, %c2} -> %arg1 as tensor<?x4xf32>{%c3}
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorTrace
//  CHECK-SAME: (%[[TENSOR0:.+]]: tensor<5xf32>, %[[TENSOR1:.+]]: tensor<?x3x?xi32>, %[[TENSOR1_DIM0:.+]]: index, %[[TENSOR1_DIM2:.+]]: index)
util.func public @tensorTrace(%tensor0: tensor<5xf32>, %tensor1: tensor<?x3x?xi32>, %tensor1_dim0: index, %tensor1_dim2: index) {
  //      CHECK: flow.tensor.trace "FOOBAR" = [
  // CHECK-SAME:   %[[TENSOR0]] : tensor<5xf32>,
  // CHECK-SAME:   %[[TENSOR1]] : tensor<?x3x?xi32>{%[[TENSOR1_DIM0]], %[[TENSOR1_DIM2]]}
  // CHECK-SAME: ]
  flow.tensor.trace "FOOBAR" = [
    %tensor0 : tensor<5xf32>,
    %tensor1 : tensor<?x3x?xi32>{%tensor1_dim0, %tensor1_dim2}
  ]
  util.return
}
