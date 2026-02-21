// RUN: iree-opt %s | FileCheck %s
//
// This file tests the missing_split_input_file warning.
// It has // ----- delimiters but no --split-input-file flag.

// CHECK-LABEL: @test1
util.func @test1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @test2
util.func @test2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}
