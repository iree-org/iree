// Companion bundle for `link_modules.mlir` test.
// Contains all modules used by the linking tests.

// Simple module with basic arithmetic.
module @simple {
  util.func public @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    util.return %result : i32
  }

  util.func public @multiply(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.muli %arg0, %arg1 : i32
    util.return %result : i32
  }
}

// Module for testing transitive dependencies: compute -> helper -> multiply_by_two.
module @transitive {
  util.func private @multiply_by_two(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %result = arith.muli %arg0, %c2 : i32
    util.return %result : i32
  }

  util.func private @helper(%arg0: i32) -> i32 {
    %0 = util.call @multiply_by_two(%arg0) : (i32) -> i32
    %c1 = arith.constant 1 : i32
    %result = arith.addi %0, %c1 : i32
    util.return %result : i32
  }

  util.func public @compute(%arg0: i32) -> i32 {
    %0 = util.call @helper(%arg0) : (i32) -> i32
    util.return %0 : i32
  }
}

// Module for testing selective linking.
module @multi {
  util.func public @func1(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %arg0, %c1 : i32
    util.return %result : i32
  }

  util.func public @func2(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %result = arith.muli %arg0, %c2 : i32
    util.return %result : i32
  }

  // This function should NOT be linked unless explicitly referenced.
  util.func public @func3(%arg0: i32) -> i32 {
    %c3 = arith.constant 3 : i32
    %result = arith.subi %arg0, %c3 : i32
    util.return %result : i32
  }
}

// Module with SCF operations.
module @with_scf {
  util.func private @process_element(%arg0: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %result = arith.addi %arg0, %c10 : i32
    util.return %result : i32
  }

  util.func public @loop_compute(%arg0: i32, %arg1: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %arg1 step %c1 iter_args(%acc = %arg0) -> (i32) {
      %0 = util.call @process_element(%acc) : (i32) -> i32
      scf.yield %0 : i32
    }
    util.return %result : i32
  }
}

// Named module A with private symbol for shadowing tests.
module @module_a {
  util.func private @internal_helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    util.return %arg0 : tensor<4xf32>
  }

  util.func public @compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = util.call @internal_helper(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    util.return %0 : tensor<4xf32>
  }

  util.func public @transform(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    util.return %arg0 : tensor<4xf32>
  }

  // Private function that conflicts with module_b's scale_factor.
  // This one returns 100.
  util.func private @scale_factor() -> i32 {
    %c100 = arith.constant 100 : i32
    util.return %c100 : i32
  }

  // Function that calls scale_factor to test shadowing.
  util.func public @compute_scaled(%arg0: i32) -> i32 {
    %0 = util.call @scale_factor() : () -> i32
    %1 = arith.muli %arg0, %0 : i32
    util.return %1 : i32
  }
}

// Named module B with conflicting private symbol for shadowing tests.
module @module_b {
  util.func public @process(%arg0: f32) -> f32 {
    %c = arith.constant 3.14 : f32
    %result = arith.addf %arg0, %c : f32
    util.return %result : f32
  }

  util.func public @validate(%arg0: f32) -> i1 {
    %c0 = arith.constant 0.0 : f32
    %result = arith.cmpf ogt, %arg0, %c0 : f32
    util.return %result : i1
  }

  // Private function that conflicts with module_a's scale_factor.
  // This one returns 200.
  util.func private @scale_factor() -> i32 {
    %c200 = arith.constant 200 : i32
    util.return %c200 : i32
  }

  // Function that calls scale_factor to test shadowing.
  util.func public @process_scaled(%arg0: i32) -> i32 {
    %0 = util.call @scale_factor() : () -> i32
    %1 = arith.addi %arg0, %0 : i32
    util.return %1 : i32
  }
}

// Module C for testing three-way conflict resolution.
module @module_c {
  util.func public @divide(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.divsi %arg0, %arg1 : i32
    util.return %result : i32
  }

  // Private function that conflicts with module_a and module_b's scale_factor.
  // This one returns 300.
  util.func private @scale_factor() -> i32 {
    %c300 = arith.constant 300 : i32
    util.return %c300 : i32
  }

  // Function that calls scale_factor to test three-way shadowing.
  util.func public @divide_scaled(%arg0: i32, %arg1: i32) -> i32 {
    %0 = util.call @scale_factor() : () -> i32
    %1 = arith.divsi %arg0, %arg1 : i32
    %2 = arith.muli %1, %0 : i32
    util.return %2 : i32
  }
}

// Module for testing globals and initializers (variant A).
// Has a mutable global @state with initial value 0.0 and initializer setting to 1.0.
module @globals_a {
  util.global private mutable @state = dense<0.0> : tensor<f32>

  util.initializer {
    %c = arith.constant dense<1.0> : tensor<f32>
    util.global.store %c, @state : tensor<f32>
    util.return
  }

  util.func public @get_state() -> tensor<f32> {
    %0 = util.global.load @state : tensor<f32>
    util.return %0 : tensor<f32>
  }
}

// Module for testing globals and initializers (variant B).
// Has a mutable global @state with initial value 0.0 and initializer setting to 2.0.
// This conflicts with globals_a's @state (same name, same initial value, different initializer).
module @globals_b {
  util.global private mutable @state = dense<0.0> : tensor<f32>

  util.initializer {
    %c = arith.constant dense<2.0> : tensor<f32>
    util.global.store %c, @state : tensor<f32>
    util.return
  }

  util.func public @get_state() -> tensor<f32> {
    %0 = util.global.load @state : tensor<f32>
    util.return %0 : tensor<f32>
  }
}

// Module for testing globals and initializers (variant C).
// Has a mutable global @state with initial value 5.0 and initializer setting to 3.0.
// This conflicts with globals_a/b's @state (same name, different initial value).
module @globals_c {
  util.global private mutable @state = dense<5.0> : tensor<f32>

  util.initializer {
    %c = arith.constant dense<3.0> : tensor<f32>
    util.global.store %c, @state : tensor<f32>
    util.return
  }

  util.func public @get_state() -> tensor<f32> {
    %0 = util.global.load @state : tensor<f32>
    util.return %0 : tensor<f32>
  }
}
