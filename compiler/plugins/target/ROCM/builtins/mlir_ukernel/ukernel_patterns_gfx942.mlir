// RUN: iree-opt -allow-unregistered-dialect %s

// F8 Patterns

// This pattern matches a medium-sized expanded matmul-like operation and
// annotates it with ukernel descriptor and configuration attributes.
pdl.pattern @annotate_matmul_like_f8_medium_expanded : benefit(1) {
  %elemtypes = pdl.attribute = [f8E4M3FNUZ, f8E4M3FNUZ, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexin maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 128 == 0, K % 128 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c128 = pdl.attribute = 128
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // N >= 1024, K >= 512
  %c512 = pdl.attribute = 512
  %c1024 = pdl.attribute = 1024
  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c512, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_medium_f8_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{
        workgroup = [1, 128, 256, 0]
      }>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_f8.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// This pattern matches a large expanded f8 matmul-like operation and annotates it
// with ukernel descriptor and configuration attributes. This is preferred over the
// medium-sized ukernel.
pdl.pattern @annotate_matmul_like_f8_large_expanded : benefit(2) {
  %elemtypes = pdl.attribute = [f8E4M3FNUZ, f8E4M3FNUZ, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 256 == 0, K % 128 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c128 = pdl.attribute = 128
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // N >= 1024, K >= 512
  %c512 = pdl.attribute = 512
  %c1024 = pdl.attribute = 1024

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c4, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c512, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_large_f8_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{
        workgroup = [1, 256, 256, 0]
      }>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_f8.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// F16 Patterns

// This pattern matches a large f16 matmul-like operation and annotates it
// with ukernel descriptor and configuration attributes.
pdl.pattern @annotate_matmul_like_f16_large : benefit(1) {
  %elemtypes = pdl.attribute = [f16, f16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c1024 = pdl.attribute = 1024
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c1, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_large_f16", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_f16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// This pattern matches a medium-sized f16 matmul-like operation and annotates it
// with ukernel descriptor and configuration attributes.
pdl.pattern @annotate_matmul_like_f16_medium_expanded : benefit(1) {
  %elemtypes = pdl.attribute = [f16, f16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 128 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c128 = pdl.attribute = 128
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c1024 = pdl.attribute = 1024

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c4, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_medium_f16_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 128, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_f16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// This pattern matches a medium-sized f16 matmul-like operation and annotates it
// with ukernel descriptor and configuration attributes. This is preferred over the
// medium-sized ukernel.
pdl.pattern @annotate_matmul_like_f16_large_expanded : benefit(2) {
  %elemtypes = pdl.attribute = [f16, f16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c1024 = pdl.attribute = 1024

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c4, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_large_f16_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 128, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_f16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// BF16 Patterns

// This pattern matches a bf16 matmul-like operation and annotates it
// with ukernel descriptor and configuration attributes.
pdl.pattern @annotate_matmul_like_bf16_large : benefit(1) {
  %elemtypes = pdl.attribute = [bf16, bf16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c1024 = pdl.attribute = 1024
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c1, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_large_bf16", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [256, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_bf16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// This pattern matches an expanded bf16 matmul-like operation of medium size and annotates it
// with ukernel descriptor and configuration attributes.
pdl.pattern @annotate_matmul_like_bf16_medium_expanded : benefit(1) {
  %elemtypes = pdl.attribute = [bf16, bf16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 128 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c128 = pdl.attribute = 128
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c128 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c4 = pdl.attribute = 4
  %c512 = pdl.attribute = 512
  %c1024 = pdl.attribute = 1024

  // TODO: Kernel specialization is needed to apply this strategy selectively at
  // runtime. Additionally model exports don't specify lower bounds so it is
  // impossible to use this strategy with this check.
  // pdl.apply_native_constraint "dimIsBound"(%lhs, %c0, %c4, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_medium_bf16_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 128, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_bf16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

// This pattern matches an expanded bf16 matmul-like operation of large size and annotates it
// with ukernel descriptor and configuration attributes. This is preferred over the medium
// strategy.
pdl.pattern @annotate_matmul_like_bf16_large_expanded : benefit(2) {
  %elemtypes = pdl.attribute = [bf16, bf16, f32]
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ]

  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %out_type = pdl.type

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %out_init = pdl.operand : %out_type

  // Match the a matmul-like generic with above indexing maps.
  %generic_op = pdl.operation (%lhs, %rhs, %out_init : !pdl.value, !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
  pdl.apply_native_constraint "matchContraction"(
        %generic_op, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  %attr_name = pdl.attribute = "iree_codegen.ukernel"
  pdl.apply_native_constraint "hasAttr"(%generic_op, %attr_name : !pdl.operation, !pdl.attribute) {isNegated = true}

  // M % 256 == 0, K % 64 == 0, N % 256 == 0
  %empty = pdl.attribute = {}
  %c0 = pdl.attribute = 0
  %c1 = pdl.attribute = 1
  %c2 = pdl.attribute = 2
  %c64 = pdl.attribute = 64
  %c256 = pdl.attribute = 256
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c1, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%lhs, %c2, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c0, %c256 : !pdl.value, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsMultipleOf"(%rhs, %c1, %c64 : !pdl.value, !pdl.attribute, !pdl.attribute)

  // M, N >= 1024, K >= 256
  %c1024 = pdl.attribute = 1024
  pdl.apply_native_constraint "dimIsBound"(%lhs, %c2, %c256, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)
  pdl.apply_native_constraint "dimIsBound"(%rhs, %c0, %c1024, %empty : !pdl.value, !pdl.attribute, !pdl.attribute, !pdl.attribute)

  pdl.rewrite {
    // Call the C++ "annotateOperation" utility to add the attributes to the matched linalg.generic op.
    // This modifies the operation in-place.

    %annotation = pdl.attribute = #iree_codegen.ukernel_descriptor<"pingpong_large_bf16_expanded", tensor>
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %attr_name, %annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %config_name = pdl.attribute = "compilation_info"
    %config = pdl.attribute = #iree_codegen.compilation_info<
      lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 256, 256, 0]}>,
      translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse
        workgroup_size = [512, 1, 1] subgroup_size = 64,
        // This strategy uses the maximum amount of possible shared memory on
        // all gfx942 architectures so shared memory padding to reduce bank
        // conflicts must be disabled. Also prefetching is done manually in the
        // above and is disabled here as well.
        {gpu_pipeline_options =
          #iree_gpu.pipeline_options<
            prefetch_shared_memory = false,
            no_reduce_shared_memory_bank_conflicts = true>,
        // This strategy requires 2 waves per SIMD.
          llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>
    >
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %config_name, %config : !pdl.operation, !pdl.attribute, !pdl.attribute)

    %builtin_attr = pdl.attribute = "rocm.builtin_name"
    %builtin_annotation = pdl.attribute = "iree_uk_amdgpu_matmul_bf16.mlir"
    pdl.apply_native_rewrite "annotateOperation"(%generic_op, %builtin_attr, %builtin_annotation : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}
