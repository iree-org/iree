module @link_module_b {
  util.func public @process(%arg0: f32) -> f32 {
    %c2 = arith.constant 2.0 : f32
    %result = arith.mulf %arg0, %c2 : f32
    util.return %result : f32
  }
}
