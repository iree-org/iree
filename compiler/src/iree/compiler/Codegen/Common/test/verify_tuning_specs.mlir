// RUN: iree-opt --verify-diagnostics --split-input-file %s

module @foo_module attributes { transform.with_named_sequence } {
  func.func @baz(%arg0: i32) -> () {
    return
  }
  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.something } {
    transform.yield %arg0 : !transform.any_op
  }
  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Must take one 'any_op' (required by 'iree_codegen.tuning_spec_entrypoint')}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Must take one 'any_op' (required by 'iree_codegen.tuning_spec_entrypoint')}}
  transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Must return one 'any_op' (required by 'iree_codegen.tuning_spec_entrypoint')}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
    attributes { iree_codegen.tuning_spec_entrypoint } {
    %0 = arith.constant 0 : i32
    transform.yield %0 : i32
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Must return one 'any_op' (required by 'iree_codegen.tuning_spec_entrypoint')}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly})
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

// expected-error @+1{{Missing named sequence '__kernel_config' (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
}

// -----

// expected-error @+1{{Missing named sequence '__kernel_config' (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  func.func @__kernel_config(%arg0: i32) -> () {
    return
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  // expected-error @+1{{Missing attribute 'iree_codegen.tuning_spec_entrypoint' in named sequence '__kernel_config' (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) {
      transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
    transform.yield
  }

  // expected-error @+1{{'__kernel_config' must contain exactly one block (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
  }

  // expected-error @+1{{'__kernel_config' must contain exactly two operations (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {

      %res_a = transform.foreach_match in %arg0
        @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      %res_b = transform.foreach_match in %res_a
        @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res_b : !transform.any_op
  }
}

// -----

// expected-error @+1{{Expected one named sequence with 'iree_codegen.tuning_spec_entrypoint', but found 2 (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
        transform.yield
    }

    // expected-error @+1{{'__kernel_config' must start with 'ForeachMatchOp' (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
        transform.print {name = "Hello"}
        transform.yield %arg0 : !transform.any_op
    }
}

// -----

// expected-error @+1{{Expected one named sequence with 'iree_codegen.tuning_spec_entrypoint', but found 2 (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }

  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield
  }
}

// -----

module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op) {
    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
    -> (!transform.any_op) {
    transform.yield %op1 : !transform.any_op
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
    // expected-error @+1 {{'ForeachMatchOp' must return exactly one 'any_op' result (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res1, %res2 = transform.foreach_match in %arg0
      @match -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield %res1 : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
     // expected-error @+1 {{'ForeachMatchOp' must return exactly one 'any_op' result (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res = transform.foreach_match in %arg0
      @match -> @apply_op_config
      : (!transform.any_op) -> (f32)

    transform.yield %res : f32
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
        -> (!transform.any_op) {
        transform.yield %arg1 : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
        transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
    // expected-error @+1 {{'ForeachMatchOp' must take exactly one 'any_op' argument (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res = transform.foreach_match in %arg0, %arg0
      @match -> @apply_op_config
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op)

    transform.yield %res : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: index) -> (index) {
    transform.yield %arg : index
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: index)
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
    // expected-error @+1 {{'ForeachMatchOp' must take exactly one 'any_op' argument (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res = transform.foreach_match in %arg0
      @match -> @apply_op_config
      : (index) -> (!transform.any_op)

    transform.yield %res : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
     // expected-error @+1{{'ForeachMatchOp' must not have 'restrict_root' attribute (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res = transform.foreach_match restrict_root in %arg0
      @match -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)

    transform.yield %res : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
    transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
     // expected-error @+1{{'ForeachMatchOp' must not have 'flatten_results' attribute (required by 'iree_codegen.tuning_spec_with_default_entrypoint')}}
    %res = transform.foreach_match flatten_results in %arg0
      @match -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op)

    transform.yield %res : !transform.any_op
  }
}
