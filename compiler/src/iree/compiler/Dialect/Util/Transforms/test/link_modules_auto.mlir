// RUN: iree-opt --iree-util-link-modules="library-path=%p" %s | FileCheck %s

// Tests auto-discovery via library-path.
// Modules are automatically loaded based on symbol prefix.

// CHECK: util.func private @link_module_a.compute
util.func private @link_module_a.compute(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK: util.func private @link_module_b.process
util.func private @link_module_b.process(%arg0: f32) -> f32

// CHECK: util.func public @main
util.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = util.call @link_module_a.compute(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %c1 = arith.constant 1.0 : f32
  %1 = util.call @link_module_b.process(%c1) : (f32) -> f32
  util.return %0 : tensor<4xf32>
}
