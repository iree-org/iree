// RUN: iree-opt --split-input-file --verify-diagnostics %s

// Checks invalid values for special key entries. We don't check the error
// message because they are IREE::Codegen::LoweringConfigTilingLevelAttr
// specific. We only care if an error is produced or not.

// expected-error@+1 {{}}
#invalid_empty_config = #iree_cpu.lowering_config<{}>

// -----

// expected-error@+1 {{}}
#invalid_distribution_config = #iree_cpu.lowering_config<distribution = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_cache_parallel_config = #iree_cpu.lowering_config<cache_parallel = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_cache_reduction_config = #iree_cpu.lowering_config<cache_reduction = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_common_parallel_config = #iree_cpu.lowering_config<vector_common_parallel = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_reduction_config = #iree_cpu.lowering_config<vector_reduction = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_inner_parallel_config = #iree_cpu.lowering_config<vector_inner_parallel = 128 : i64>
