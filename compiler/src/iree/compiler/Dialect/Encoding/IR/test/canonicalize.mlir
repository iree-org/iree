// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

#encoding = #iree_encoding.testing<>

// CHECK-LABEL: @foldUnsetOfSetEncoding
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x4xf32>)
util.func public @foldUnsetOfSetEncoding(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: iree_encoding.set_encoding
  // CHECK-NOT: iree_encoding.unset_encoding
  %0 = iree_encoding.set_encoding %arg0 : tensor<4x4xf32> -> tensor<4x4xf32, #encoding>
  %1 = iree_encoding.unset_encoding %0 : tensor<4x4xf32, #encoding> -> tensor<4x4xf32>
  // CHECK: util.return %[[ARG0]]
  util.return %1 : tensor<4x4xf32>
}

// -----

#encoding = #iree_encoding.testing<>

// CHECK-LABEL: @foldUnsetOfSetEncodingDynamic
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x4xf32>
util.func public @foldUnsetOfSetEncodingDynamic(%arg0: tensor<?x4xf32>, %dim: index) -> tensor<?x4xf32> {
  // CHECK-NOT: iree_encoding.set_encoding
  // CHECK-NOT: iree_encoding.unset_encoding
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x4xf32> -> tensor<?x4xf32, #encoding>
  %1 = iree_encoding.unset_encoding %0 : tensor<?x4xf32, #encoding> -> tensor<?x4xf32>{%dim}
  // CHECK: util.return %[[ARG0]]
  util.return %1 : tensor<?x4xf32>
}

// -----

#encoding = #iree_encoding.testing<>

// A round-trip that does not land back on the original type must not fold.
// CHECK-LABEL: @dontFoldUnsetOfSetEncodingDifferentType
util.func public @dontFoldUnsetOfSetEncodingDifferentType(%arg0: tensor<4x4xf32>, %d0: index, %d1: index) -> tensor<?x?xf32> {
  // CHECK: iree_encoding.set_encoding
  %0 = iree_encoding.set_encoding %arg0 : tensor<4x4xf32> -> tensor<4x4xf32, #encoding>
  // CHECK: iree_encoding.unset_encoding
  %1 = iree_encoding.unset_encoding %0 : tensor<4x4xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  util.return %1 : tensor<?x?xf32>
}

// -----

#encoding = #iree_encoding.testing<>

// An unset_encoding whose source is not a set_encoding must not fold.
// CHECK-LABEL: @dontFoldUnsetWithoutSet
util.func public @dontFoldUnsetWithoutSet(%arg0: tensor<4x4xf32, #encoding>) -> tensor<4x4xf32> {
  // CHECK: iree_encoding.unset_encoding
  %0 = iree_encoding.unset_encoding %arg0 : tensor<4x4xf32, #encoding> -> tensor<4x4xf32>
  util.return %0 : tensor<4x4xf32>
}
