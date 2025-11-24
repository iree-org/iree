// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error@+1 {{expected valid keyword or string}}
#invalid_empty_config = #iree_cpu.lowering_config<{}>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_distribution_config = #iree_cpu.lowering_config<distribution = 128 : i64>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_cache_parallel_config = #iree_cpu.lowering_config<cache_parallel = 128 : i64>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_cache_reduction_config = #iree_cpu.lowering_config<cache_reduction = 128 : i64>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_vector_common_parallel_config = #iree_cpu.lowering_config<vector_common_parallel = 128 : i64>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_vector_reduction_config = #iree_cpu.lowering_config<vector_reduction = 128 : i64>

// -----

// expected-error@+1 {{expected '{'}}
#invalid_vector_inner_parallel_config = #iree_cpu.lowering_config<vector_inner_parallel = 128 : i64>
