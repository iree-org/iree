// Module B: Leaf dependency with utility functions.

module @module_b {
  // Simple helper function that doubles an input.
  util.func public @helper(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %result = arith.muli %arg0, %c2 : i32
    util.return %result : i32
  }

  // Processing function that adds a constant.
  util.func public @process(%arg0: f32) -> f32 {
    %c = arith.constant 3.14 : f32
    %result = arith.addf %arg0, %c : f32
    util.return %result : f32
  }
}
