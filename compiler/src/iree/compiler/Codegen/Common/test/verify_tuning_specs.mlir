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
  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
  transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
    attributes { iree_codegen.tuning_spec_entrypoint } {
    %0 = arith.constant 0 : i32
    transform.yield %0 : i32
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly})
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

// expected-error @+1{{The tuning specification must include a named sequence with the symbol name '__kernel_config'}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
}

// -----

// expected-error @+1{{The tuning specification must include a named sequence with the symbol name '__kernel_config'}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  func.func @__kernel_config(%arg0: i32) -> () {
    return
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) {
      transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
  }

  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {

      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res_b : !transform.any_op
  }
}

// -----

// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
  }

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
  }

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {

      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
  }

  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.print {name = "Hello Tuning Spec", skip_regions}
    transform.yield %arg0 : !transform.any_op
  }
}
