// RUN: iree-opt --split-input-file --verify-diagnostics --iree-codegen-link-tuning-specs %s | FileCheck %s

// Check test cases where tuning specs with default attributes fail to merge and fall back to linking all the tuning specs,
// verify the emitted warning messages.

module @test_module_foreach_two_args attributes { transform.with_named_sequence } {
  module @inner_module_a
  attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
    transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
        -> (!transform.any_op) {
        transform.yield %arg1 : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
        transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
        %extra_arg = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
        // expected-warning @+1 {{ForeachMatchOp must take exactly one any_op argument.}}
        %res = transform.foreach_match in %arg0, %extra_arg @match -> @apply_op_config
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op)

        transform.yield %res : !transform.any_op
    }
  }

  module @inner_module_b
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
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
  }
}

// CHECK-LABEL:   module @test_module_foreach_two_args
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           inner_module_a::@__kernel_config_1
// CHECK:           inner_module_b::@__kernel_config_2

// -----

module @test_module_two_results attributes { transform.with_named_sequence } {
  module @inner_module_a
  attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
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
    // expected-warning @+1 {{ForeachMatchOp must return exactly one any_op result.}}
    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield %res1 : !transform.any_op
  }
}

  module @inner_module_b
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
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
  }
}

// CHECK-LABEL:   module @test_module_two_results
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           inner_module_a::@__kernel_config_1
// CHECK:           inner_module_b::@__kernel_config_2

// -----

module @test_module_mismatched_restrict_root attributes { transform.with_named_sequence } {
  module @inner_module_a
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      // expected-warning @+1 {{Mismatched 'restrict_root' attributes across ForeachMatchOps.}}
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }

  module @inner_module_b
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match restrict_root in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }
}

// CHECK-LABEL:   module @test_module_mismatched_restrict_root
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           inner_module_a::@__kernel_config_1
// CHECK:           inner_module_b::@__kernel_config_2

// -----

module @test_module_mismatched_flatten_results attributes { transform.with_named_sequence } {
  module @inner_module_a
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      // expected-warning @+1 {{Mismatched 'flatten_results' attributes across ForeachMatchOps.}}
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }

  module @inner_module_b
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match flatten_results in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }
}

// CHECK-LABEL:   module @test_module_mismatched_flatten_results
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           inner_module_a::@__kernel_config_1
// CHECK:           inner_module_b::@__kernel_config_2
