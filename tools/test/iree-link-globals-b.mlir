// Test module with globals and initializers (variant B).
module {
  util.global private mutable @state = dense<0.0> : tensor<f32>

  util.initializer {
    %c = arith.constant dense<2.0> : tensor<f32>
    util.global.store %c, @state : tensor<f32>
    util.return
  }

  util.func private @get_state() -> tensor<f32> {
    %0 = util.global.load @state : tensor<f32>
    util.return %0 : tensor<f32>
  }
}
