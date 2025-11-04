// Module A: Mid-level dependency with flow.executable and transitive dep on B.

module @module_a {
  // External reference to module B (transitive dependency).
  util.func private @module_b.process(%arg0: f32) -> f32

  // Compute function that uses a dispatch and calls module B.
  util.func public @compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // Dispatch to the executable.
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch @compute_dispatch_0::@compute_dispatch_0_elementwise_4_f32[%c1, %c1, %c1](%arg0) : (tensor<4xf32>) -> tensor<4xf32>

    // Call module B for post-processing (transitive dependency).
    %c1_f32 = arith.constant 1.0 : f32
    %processed = util.call @module_b.process(%c1_f32) : (f32) -> f32

    util.return %0 : tensor<4xf32>
  }

  // Flow executable for the dispatch (simplified for testing).
  flow.executable private @compute_dispatch_0 {
    flow.executable.export public @compute_dispatch_0_elementwise_4_f32 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      flow.return %arg0, %arg1, %arg2 : index, index, index
    }
    builtin.module {
      func.func @compute_dispatch_0_elementwise_4_f32() {
        return
      }
    }
  }
}
