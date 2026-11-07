// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error@+1 {{stream.min_access_granularity must be a positive power of two representable in i64}}
#invalid = #hal.device.target<"word", {stream.min_access_granularity = 0 : i64}>

// -----

// expected-error@+1 {{stream.min_fill_granularity must be a positive power of two representable in i64}}
#invalid = #hal.device.target<"word", {stream.min_fill_granularity = -4 : i64}>

// -----

// expected-error@+1 {{stream.min_copy_granularity must be a positive power of two representable in i64}}
#invalid = #hal.device.target<"word", {stream.min_copy_granularity = 3 : i64}>

// -----

// expected-error@+1 {{stream.min_access_granularity must be a positive power of two representable in i64}}
#invalid = #hal.device.target<"word", {stream.min_access_granularity = "word"}>

// -----

// expected-error@+1 {{stream.min_fill_granularity must be a positive power of two representable in i64}}
#invalid = #hal.device.target<"word", {stream.min_fill_granularity = 18446744073709551616 : i128}>
