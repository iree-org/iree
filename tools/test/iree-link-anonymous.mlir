// Input module for testing linking with anonymous modules.
// Used as data by iree-link.mlir and iree-link-bundle.mlir tests.

// External references to named modules (with prefix).
util.func private @module_a.compute(%arg0: tensor<4xf32>) -> tensor<4xf32>
util.func private @module_b.helper(%arg0: i32) -> i32

// External references to anonymous module C (no prefix).
util.func private @subtract(%arg0: i32, %arg1: i32) -> i32
util.func private @double(%arg0: f32) -> f32

// Private helper function that conflicts with module-c's private function.
// This one returns 10, module-c's returns 5.
util.func private @scale_factor() -> i32 {
  %c10 = arith.constant 10 : i32
  util.return %c10 : i32
}

// Entry point that uses functions from named and anonymous modules.
util.func public @test_anonymous(%arg0: tensor<4xf32>, %arg1: i32, %arg2: f32) -> (tensor<4xf32>, i32, f32, i32) {
  // Use named module functions.
  %0 = util.call @module_a.compute(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = util.call @module_b.helper(%arg1) : (i32) -> i32

  // Use anonymous module's functions (no prefix).
  %2 = util.call @subtract(%arg1, %arg1) : (i32, i32) -> i32
  %3 = util.call @double(%arg2) : (f32) -> f32

  // Use our private helper (should get our version that returns 10).
  %4 = util.call @scale_factor() : () -> i32

  util.return %0, %2, %3, %4 : tensor<4xf32>, i32, f32, i32
}
