// RUN: iree-opt %s -split-input-file -verify-diagnostics

#encoding = #iree_encoding.testing<>
util.func public @tensorBitCastWithDiffEncoding(%arg0 : tensor<16xi32, #encoding>) -> tensor<16xi32> {
  // expected-error@+1 {{the source and result of a bitcast should have the same encoding}}
  %0 = flow.tensor.bitcast %arg0 : tensor<16xi32, #encoding> -> tensor<16xi32>
  util.return %0 : tensor<16xi32>
}
