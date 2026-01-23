// Module C: Anonymous module (no module name) with utility functions.

module {
  // Private helper that conflicts with iree-link-anonymous.mlir's private function.
  // This one returns 5, the main module's returns 10.
  // Tests that mergeModuleInto properly shadows private symbols.
  util.func private @scale_factor() -> i32 {
    %c5 = arith.constant 5 : i32
    util.return %c5 : i32
  }

  // Utility function that subtracts values.
  util.func public @subtract(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.subi %arg0, %arg1 : i32
    util.return %result : i32
  }

  // Utility function that multiplies a float by 2.
  // Uses the private helper to test it gets pulled in.
  util.func public @double(%arg0: f32) -> f32 {
    %c2 = arith.constant 2.0 : f32
    %result = arith.mulf %arg0, %c2 : f32

    // Call private helper to ensure it's included in the link.
    %scale = util.call @scale_factor() : () -> i32

    util.return %result : f32
  }
}
