// RUN: iree-opt %s

// This is just an initial tuning spec for gfx942 and is not intended for
// production use.
// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
// configurations to this spec.

module @iree_default_tuning_spec_gfx942 attributes { transform.with_named_sequence } {

transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.readonly}) -> ()
  attributes { iree_codegen.tuning_spec_entrypoint } {
  transform.yield
}

}
