// Test module with globals and initializers (variant C).
module {
  util.global private mutable @state = dense<5.0> : tensor<f32>

  util.initializer {
    %c = arith.constant dense<3.0> : tensor<f32>
    util.global.store %c, @state : tensor<f32>
    util.return
  }

  util.func private @get_state() -> tensor<f32> {
    %0 = util.global.load @state : tensor<f32>
    util.return %0 : tensor<f32>
  }
}
