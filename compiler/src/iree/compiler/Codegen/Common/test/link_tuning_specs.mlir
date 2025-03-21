// RUN: iree-opt %s --no-implicit-module --iree-codegen-link-tuning-specs --split-input-file \
// RUN:   | FileCheck %s

// CHECK-LABEL: module @td_module_0
//
// CHECK:         transform.named_sequence @outer_spec
//
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           %[[OP1:.+]] = transform.include @foo_module::@foo failures(suppress) (%arg0)
// CHECK-NEXT:      %[[OP2:.+]] = transform.include @bar_module::@bar failures(suppress) (%[[OP1]])
// CHECK-NEXT:      %[[OP3:.+]] = transform.include @baz_module::@baz failures(suppress) (%[[OP2]])
// CHECK-NEXT:      transform.yield %[[OP3]] : !transform.any_op

module @td_module_0 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.print {name = "Foo", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  module @bar_module attributes { transform.with_named_sequence } {
    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
      transform.print {name = "Bar", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  module @baz_module attributes { transform.with_named_sequence } {
    transform.named_sequence @baz(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.print {name = "Baz", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  transform.named_sequence @outer_spec(%module: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %module : !transform.any_op
  }
}


// -----

// Here, `foo` shouldn't be included because it's not marked with `tuning_spec_entrypoint`.

// CHECK-LABEL: module @td_module_1
// CHECK:       @foo_module
// CHECK:       @__kernel_config(
// CHECK-NOT      transform.include @foo_module::@foo failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK:         transform.include @foo_module::@bar failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:    transform.yield

module @td_module_1 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}


// -----

// Make sure we do not crash on modules with no tuning specs.

// CHECK-LABEL: module @td_module_2
// CHECK-NOT:   @__kernel_config
module @td_module_2 attributes { transform.with_named_sequence } {}

// -----

// Make sure we do not crash on unnamed nested modules.

// CHECK-LABEL: module @td_module_3
// CHECK:       transform.named_sequence @foo
// CHECK-NOT:     @__kernel_config

module @td_module_3 attributes { transform.with_named_sequence } {
  module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
}

// -----

// Make sure that the names of all included specs and the outermost entrypoint
// are kept unique.

// CHECK-LABEL: module @td_module_4
// CHECK:       @foo_module attributes
// CHECK:       @bar_module attributes
// CHECK:       @__kernel_config(
// CHECK:         transform.include @foo_module::@foo failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK:         transform.include @foo_module::@__kernel_config_1 failures(suppress)
// CHECK:         transform.include @bar_module::@foo_1 failures(suppress)
// CHECK:         transform.include @bar_module::@__kernel_config_2 failures(suppress)
// CHECK-NEXT:    transform.yield

module @td_module_4 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
  module @bar_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
}

// -----

// make sure the inner module are flatting out and foreach_match ops are merged.

module @td_module attributes { transform.with_named_sequence } {
    module @mmt_module attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
        transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                                %config: !transform.any_param {transform.readonly}) {
        transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
        transform.yield
        }

        transform.named_sequence @match_mmt_f16_f16_f32(%matmul: !transform.any_op {transform.readonly})
            -> (!transform.any_op, !transform.any_param) {
        transform.match.operation_name %matmul ["linalg.generic"] : !transform.any_op
        %config = transform.param.constant {key = "custom_config"} -> !transform.any_param
        transform.yield %matmul, %config : !transform.any_op, !transform.any_param
        }

        transform.named_sequence
        @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
        attributes { iree_codegen.tuning_spec_entrypoint } {
        %res = transform.foreach_match in %variant_op
            @match_mmt_f16_f16_f32 -> @apply_op_config
            : (!transform.any_op) -> !transform.any_op
        transform.yield %res : !transform.any_op
        }
    }

    module @attention_module attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
        transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                 %config: !transform.any_param {transform.readonly},
                                                 %decomposition_config: !transform.any_param {transform.readonly}) {
            transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
            transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
            transform.yield
        }

        transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
            transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
            %config = transform.param.constant {key = "attn_config"} -> !transform.any_param
            %decomposition_config = transform.param.constant {key = "decomp_config"} -> !transform.any_param
            transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
        }

        transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
        attributes { iree_codegen.tuning_spec_entrypoint } {
            %res = transform.foreach_match in %variant_op
            @match_attention_f16 -> @apply_attn_op_config
            : (!transform.any_op) -> !transform.any_op
            transform.yield %res : !transform.any_op
        }
    }
}

// CHECK-LABEL:   module @td_module
// CHECK-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
// CHECK-NOT:     @mmt_module
// CHECK-NOT:     @attention_module
// CHECK:         @__kernel_config(
// CHECK:         transform.foreach_match
// CHECK:         @match_mmt_f16_f16_f32 -> @apply_op_config
// CHECK-NEXT:    @match_attention_f16 -> @apply_attn_op_config
// CHECK-NEXT:    transform.yield

// -----

// Make sure that all named sequence operation names in the merged foreach_match remain unique.

module @td_module attributes { transform.with_named_sequence } {
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
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }

  module @inner_module_c
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @apply_op_config_1(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config_1
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }

    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }
  }
}

// CHECK-LABEL:   module @td_module
// CHECK-NOT:     module @inner_module_a
// CHECK-NOT:     module @inner_module_b
// CHECK-NOT:     module @inner_module_c
// CHECK:         @__kernel_config(
// CHECK:         transform.foreach_match
// CHECK:           @match -> @apply_op_config
// CHECK:           @inner_module_b_match -> @inner_module_b_apply_op_config
// CHECK:           @inner_module_c_match -> @apply_op_config_1

// -----

// Make sure that all named sequence operation names in the merged foreach_match remain unique even when inner module names are missing.

module @td_module attributes { transform.with_named_sequence } {
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
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }

  module
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

  module
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config_1(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config_1
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }
}

// CHECK-LABEL:   module @td_module
// CHECK-NOT:     module @inner_module_a
// CHECK-NOT:     module @inner_module_b
// CHECK-NOT:     module @inner_module_c
// CHECK:         @__kernel_config(
// CHECK:         transform.foreach_match
// CHECK:           @match -> @apply_op_config
// CHECK:           @m0_match -> @m0_apply_op_config
// CHECK:           @m1_match -> @apply_op_config_1
