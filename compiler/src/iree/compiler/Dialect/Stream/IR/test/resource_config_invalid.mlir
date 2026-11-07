// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error @+2 {{granularity 3 is not a power of two}}
util.func private @granularity_not_power_of_two() attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 64, max_buffer_range = 2048, min_buffer_range_alignment = 16, index_bits = 32, min_fill_granularity = 3}>
}

// -----

// expected-error @+2 {{min_buffer_offset_alignment 16 is not a multiple of granularity 32}}
util.func private @granularity_exceeds_offset_alignment() attributes {
  stream.resources = #stream.resource_config<{max_allocation_size = 4096, min_buffer_offset_alignment = 16, max_buffer_range = 2048, min_buffer_range_alignment = 16, index_bits = 32, min_access_granularity = 32}>
}

// -----

// expected-error@+2 {{granularity 0 is not a power of two}}
util.func private @invalid_min_access_granularity() attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 64, min_buffer_range_alignment = 16, min_access_granularity = 0}>
}

// -----

// expected-error@+2 {{granularity -2 is not a power of two}}
util.func private @invalid_min_copy_granularity() attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 64, min_buffer_range_alignment = 16, min_copy_granularity = -2}>
}

// -----

// expected-error@+2 {{min_buffer_range_alignment 16 is not a multiple of granularity 32}}
util.func private @invalid_min_fill_granularity() attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 64, min_buffer_range_alignment = 16, min_fill_granularity = 32}>
}
