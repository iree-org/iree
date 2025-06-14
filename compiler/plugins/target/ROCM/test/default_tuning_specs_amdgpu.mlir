// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --no-implicit-module %s | FileCheck %s --check-prefix=DEFAULT

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec_mmt_tile_and_fuse.mlir \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --no-implicit-module %s | FileCheck %s --check-prefix=BOTH

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec_mmt_tile_and_fuse_default.mlir \
// RUN:   --iree-codegen-enable-default-tuning-specs \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --no-implicit-module %s | FileCheck %s --check-prefix=MERGE

// Note: This test needs to be in the plugin subdirectory because it depends
// on the default spec that's only embedded in the compiler library when the
// ROCM plugin is built.

// ============================================================================

// Check that the default tuning spec gets materialized without linking.

// DEFAULT-LABEL: module @iree_default_tuning_spec_gfx942
// DEFAULT-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
// DEFAULT-SAME:    transform.with_named_sequence
// DEFAULT-LABEL:   transform.named_sequence @__kernel_config
// DEFAULT-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}

// Check that the default tuning spec gets materialized as a module attribute.
// DEFAULT:        module attributes
// DEFAULT-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// DEFAULT-LABEL:    func.func @main_0

// ============================================================================

// Check that both the user tuning spec and the default spec get linked and
// materialized. The user spec should have precedence over the default one.

// TODO: Re-add the check for iree_codegen.tuning_spec_with_default_entrypoint
// once new linking is added and the output IR can pass verification for the default attribute.

// BOTH-LABEL: module @iree_linked_tuning_spec
// BOTH-SAME:    transform.with_named_sequence
// BOTH-LABEL:   module @mmt_tile_and_fuse_spec_0 attributes {transform.with_named_sequence}
// BOTH-LABEL:     transform.named_sequence @main
// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// BOTH-LABEL:   module @iree_default_tuning_spec_gfx942_1 attributes {transform.with_named_sequence}
// BOTH:           transform.named_sequence @__kernel_config
// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// BOTH:         transform.named_sequence @__kernel_config
// BOTH:           @mmt_tile_and_fuse_spec_0::@main
// BOTH:           @iree_default_tuning_spec_gfx942_1::@__kernel_config

// BOTH:        module attributes
// BOTH-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// BOTH-LABEL:    func.func @main_0

// ============================================================================

// Check that both the user tuning spec and the default spec get merged and
// materialized, in which nested structure should not be present, and a merged
// foreach_match op should exist. The user spec should have precedence over the
// default one.

// MERGE:       module @iree_linked_tuning_spec
// MERGE-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
// MERGE-SAME:    transform.with_named_sequence
// MERGE-NOT:     module @mmt_tile_and_fuse_spec
// MERGE-NOT:     module @iree_default_tuning_spec_gfx942
// MERGE:         transform.named_sequence @__kernel_config
// MERGE-SAME:    attributes {iree_codegen.tuning_spec_entrypoint}
// MERGE:         transform.foreach_match
// MERGE:           @match_mmt -> @apply_op_config,
// MERGE-NEXT:      @match_mmt_f16_f16_f32_large_expanded -> @apply_expanded_pingpong_op_config,
// MERGE-NEXT:      @match_mmt_f8_f8_f32_large_expanded -> @apply_expanded_f8_pingpong_op_config,
// MERGE-NEXT:      @match_mmt_f16_f16_f32_large -> @apply_pingpong_op_config,
// MERGE-NEXT:      @match_mmt_f16_f16_f32_medium_expanded -> @apply_expanded_medium_pingpong_op_config,
// MERGE-NEXT:      @match_mmt_f8_f8_f32_medium_expanded -> @apply_expanded_f8_medium_pingpong_op_config,
// MERGE-NEXT:      @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config,
// MERGE-NEXT:      @match_mmt_2048x1280x5120_f16_f16_f32 -> @iree_default_tuning_spec_gfx942_1_apply_op_config

// NOTE: The order matters above because `foreach_match` ops performs matching from top to bottom.

// MERGE:        module attributes
// MERGE-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// MERGE-LABEL:    func.func @main_0

module attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
    iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
  }>
} {
  func.func @main_0() {
    return
  }
}
