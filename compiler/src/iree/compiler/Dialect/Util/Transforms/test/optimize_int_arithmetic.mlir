// RUN: iree-opt --split-input-file --iree-util-optimize-int-arithmetic  %s | FileCheck %s
// We inherit a number of patterns from upstream for optimizing specific arith
// operations. Those are not the focus of testing, but we may test some of them
// here incidentally as part of verifying that the overall pass and local
// patterns are effective.
// Many of these tests take advantage of the fact that if a value can be
// inferred for arith.cmpi, a constant i1 will be substituted for it.

// CHECK-LABEL: @index_upper_bound
util.func @index_upper_bound(%arg0 : index) -> i1 {
  // CHECK: %[[RESULT:.*]] = arith.constant true
  // CHECK: util.return %[[RESULT]]
  %cst = arith.constant 101 : index
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.cmpi ult, %0, %cst : index
  util.return %1 : i1
}

// -----
// CHECK-LABEL: @index_lower_bound
util.func @index_lower_bound(%arg0 : index) -> i1 {
  // CHECK: %[[RESULT:.*]] = arith.constant true
  // CHECK: util.return %[[RESULT]]
  %cst = arith.constant 5 : index
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.cmpi ugt, %0, %cst : index
  util.return %1 : i1
}

// -----
// If there is a missing umax in a multi-row assumption, then it must
// be treated as having no known upper bound.
// CHECK-LABEL: @missing_umax_skipped
util.func @missing_umax_skipped(%arg0 : index) -> i1 {
  // CHECK: arith.cmpi
  %cst = arith.constant 101 : index
  %0 = util.assume.int %arg0[<umin=10, umax=100>, <umin=10>] : index
  %1 = arith.cmpi ult, %0, %cst : index
  util.return %1 : i1
}

// -----
// If there is a missing umin in a multi-row assumption, then it must
// be treated as having no known lower bound.
// CHECK-LABEL: @missing_umin_skipped
util.func @missing_umin_skipped(%arg0 : index) -> i1 {
  // CHECK: arith.cmpi
  %cst = arith.constant 5 : index
  %0 = util.assume.int %arg0[<umin=10, umax=100>, <umax=100>] : index
  %1 = arith.cmpi ugt, %0, %cst : index
  util.return %1 : i1
}

// -----
// CHECK-LABEL: @index_indeterminate
util.func @index_indeterminate(%arg0 : index) -> i1 {
  // CHECK: arith.cmpi
  %cst = arith.constant 50 : index
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.cmpi ugt, %0, %cst : index
  util.return %1 : i1
}

// -----
// CHECK-LABEL: @index_multi_assumptions_unioned
util.func @index_multi_assumptions_unioned(%arg0 : index) -> i1, i1, i1 {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  // CHECK-DAG: %[[C51:.*]] = arith.constant 51
  %cst5 = arith.constant 5 : index
  %cst51 = arith.constant 51 : index
  %cst101 = arith.constant 101 : index
  %0 = util.assume.int %arg0[
      <umin=10, umax=100>,
      <umin=5, umax=50>
   ] : index
  %1 = arith.cmpi ult, %0, %cst5 : index   // Statically false
  // CHECK: %[[DYNAMIC:.*]] = arith.cmpi ult, {{.*}}, %[[C51]]
  %2 = arith.cmpi ult, %0, %cst51 : index  // Cannot be determined
  %3 = arith.cmpi ult, %0, %cst101 : index // Statically true
  // CHECK: return %[[FALSE]], %[[DYNAMIC]], %[[TRUE]]
  util.return %1, %2, %3 : i1, i1, i1
}

// -----
// This checks a corner case that has to line up with how util.assume.int
// signals its range to the int range analsysis. Here, if interpreting the
// umax as a signed value (which is what is used for evaluating an sgt in the
// arith.cmpi op), it poisons the analysis by assuming that the signed range is
// the entire signed range of the data type. This means that any signed
// evaluation will fail, whereas an unsigned will succeed.
// CHECK-LABEL: @index_unsigned_overflow_signed
util.func @index_unsigned_overflow_signed(%arg0 : index) -> i1, i1 {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  %cst = arith.constant 5 : index
  %0 = util.assume.int %arg0<umin=10, umax=18446744073709551615> : index
  // CHECK: %[[POISON:.*]] = arith.cmpi sgt
  %1 = arith.cmpi sgt, %0, %cst : index
  %2 = arith.cmpi ugt, %0, %cst : index
  // CHECK: util.return %[[POISON]], %[[TRUE]]
  util.return %1, %2 : i1, i1
}

// -----
// Minimal testing to ensure that integer data types < 64 bits do the right
// thing. This exercises some APInt bit manipulation in our interface
// implementations.
// CHECK-LABEL: @int_upper_bound
util.func @int_upper_bound(%arg0 : i32) -> i1 {
  // CHECK: %[[RESULT:.*]] = arith.constant true
  // CHECK: util.return %[[RESULT]]
  %cst = arith.constant 101 : i32
  %0 = util.assume.int %arg0<umin=10, umax=100> : i32
  %1 = arith.cmpi ult, %0, %cst : i32
  util.return %1 : i1
}

// -----
// Validate the signed/unsigned mismatch corner case on a type narrower than
// 64 bits.
// CHECK-LABEL: @int32_unsigned_overflow_signed
util.func @int32_unsigned_overflow_signed(%arg0 : i32) -> i1, i1 {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  %cst = arith.constant 5 : i32
  // Max is one greater than the i32 signed positive range.
  %0 = util.assume.int %arg0<umin=10, umax=2147483648> : i32
  // CHECK: %[[POISON:.*]] = arith.cmpi sgt
  %1 = arith.cmpi sgt, %0, %cst : i32
  %2 = arith.cmpi ugt, %0, %cst : i32
  // CHECK: util.return %[[POISON]], %[[TRUE]]
  util.return %1, %2 : i1, i1
}

// -----
// CHECK-LABEL: @to_unsigned_ceildivsi
util.func @to_unsigned_ceildivsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // CHECK: ceildivui
  // CHECK: ceildivsi
  // CHECK: ceildivsi
  %2 = arith.ceildivsi %0, %0 : i64
  %3 = arith.ceildivsi %0, %1 : i64
  %4 = arith.ceildivsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_divsi
util.func @to_unsigned_divsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // CHECK: divui
  // CHECK: divsi
  // CHECK: divsi
  %2 = arith.divsi %0, %0 : i64
  %3 = arith.divsi %0, %1 : i64
  %4 = arith.divsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_floordivsi
util.func @to_unsigned_floordivsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // CHECK: divui
  // CHECK: divsi
  // CHECK: divsi
  %2 = arith.floordivsi %0, %0 : i64
  %3 = arith.floordivsi %0, %1 : i64
  %4 = arith.floordivsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_index_cast
util.func @to_unsigned_index_cast(%arg0 : index, %arg1 : index) -> i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : index
  // CHECK: index_castui
  %2 = arith.index_cast %0 : index to i64
  // CHECK: index_cast
  %3 = arith.index_cast %1 : index to i64
  util.return %2, %3 : i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_remsi
util.func @to_unsigned_remsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // CHECK: remui
  // CHECK: remsi
  // CHECK: remsi
  %2 = arith.remsi %0, %0 : i64
  %3 = arith.remsi %0, %1 : i64
  %4 = arith.remsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_minsi
util.func @to_unsigned_minsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // Note that the first is converted to unsigned and then can be elided
  // entirely.
  // CHECK-NOT: minui
  // CHECK: minsi
  // CHECK: minsi
  %2 = arith.minsi %0, %0 : i64
  %3 = arith.minsi %0, %1 : i64
  %4 = arith.minsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_maxsi
util.func @to_unsigned_maxsi(%arg0 : i64, %arg1 : i64) -> i64, i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i64
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=9223372036854775808> : i64
  // Note that the first is converted to unsigned and then can be elided
  // entirely.
  // CHECK-NOT: maxui
  // CHECK: maxsi
  // CHECK: maxsi
  %2 = arith.maxsi %0, %0 : i64
  %3 = arith.maxsi %0, %1 : i64
  %4 = arith.maxsi %1, %0 : i64
  util.return %2, %3, %4 : i64, i64, i64
}

// -----
// CHECK-LABEL: @to_unsigned_extsi
util.func @to_unsigned_extsi(%arg0 : i32, %arg1 : i32) -> i64, i64 {
  %0 = util.assume.int %arg0<umin=10, umax=100> : i32
  // One greater than the signed maximum.
  %1 = util.assume.int %arg1<umin=10, umax=2147483648> : i32
  // CHECK: extui
  %2 = arith.extsi %0 : i32 to i64
  // CHECK: extsi
  %3 = arith.extsi %1 : i32 to i64
  util.return %2, %3 : i64, i64
}

// -----
// Tests the ConvertUnsignedI64IndexCastProducerToIndex pattern and the
// composition with other patterns to collapse entire sequences of
// index_cast (signed) -> i64 -> index_cast (signed) -> index.
// This sequence of tests uses signed ops where they exist so as to ensure that
// the cascade of rewrites and additional analysis composes together. This
// specifically tests that the listener properly erases/flushes and triggers
// additional cycles.
// CHECK-LABEL: @index_cast_i64_to_index_addi
util.func @index_cast_i64_to_index_addi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.addi %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.addi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// Multi-use should not convert
// CHECK-LABEL: @index_cast_i64_to_index_addi_multiuse
util.func @index_cast_i64_to_index_addi_multiuse(%arg0 : index, %arg1 : index) -> index, i64 {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  // CHECK: arith.index_cast
  // CHECK: arith.index_cast
  %1 = arith.index_cast %0 : index to i64
  %2 = arith.addi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3, %2 : index, i64
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_ceildivsi
util.func @index_cast_i64_to_index_ceildivsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.ceildivui %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.ceildivsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_floordivsi
util.func @index_cast_i64_to_index_floordivsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.divui %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.floordivsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_maxsi
util.func @index_cast_i64_to_index_maxsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // Note that the entire sequence is inferred to be removed.
  // CHECK: util.return %[[ASSUME]]
  %2 = arith.maxsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_minsi
util.func @index_cast_i64_to_index_minsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // Note that the entire sequence is inferred to be removed.
  // CHECK: util.return %[[ASSUME]]
  %2 = arith.minsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_muli
util.func @index_cast_i64_to_index_muli(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.muli %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.muli %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_remsi
util.func @index_cast_i64_to_index_remsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.remui %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.remsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_subi
util.func @index_cast_i64_to_index_subi(%arg0 : index, %arg1 : index) -> index {
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // Note that the subtraction should be inferred as elided.
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: util.return %[[ZERO]]
  %2 = arith.subi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_addi_bad_signed_bounds
util.func @index_cast_i64_to_index_addi_bad_signed_bounds(%arg0 : i64) -> index {
  %cst1 = arith.constant 1 : i64
  // Maximum 32bit unsigned value +1 (from the addi should reject).
  %0 = util.assume.int %arg0<umin=10, umax=4294967295> : i64
  %2 = arith.addi %0, %cst1 : i64
  // Out of bounds of conservative 32bit values so do not convert.
  // CHECK: arith.addi
  // CHECK: arith.index_castui
  %3 = arith.index_castui %2 : i64 to index
  util.return %3 : index
}

// -----
// Validate the index unsigned 32bit overflow case.
// CHECK-LABEL: @index_unsigned_overflow_signed
util.func @index_unsigned_overflow_signed(%arg0 : index) -> index {
  %cst = arith.constant 5 : index
  // Max is one greater than the i32 unsigned range.
  %0 = util.assume.int %arg0<umin=10, umax=4294967296> : index
  // Should not convert to unsigned
  // CHECK: arith.divsi
  %1 = arith.divsi %0, %cst : index
  util.return %1 : index
}

// -----
// CHECK-LABEL: @index_cast_i64_to_index_remsi
util.func @index_cast_i64_to_index_remsi(%arg0 : index, %arg1 : index) -> index {
  // CHECK: %[[ASSUME:.*]] = util.assume.int
  %0 = util.assume.int %arg0<umin=10, umax=100> : index
  %1 = arith.index_cast %0 : index to i64
  // CHECK: arith.remui %[[ASSUME]], %[[ASSUME]] : index
  %2 = arith.remsi %1, %1 : i64
  %3 = arith.index_cast %2 : i64 to index
  util.return %3 : index
}

// -----
// Truncate of an index cast can be folded into the index cast.
// CHECK-LABEL: @elide_trunc_of_index_castui
util.func @elide_trunc_of_index_castui(%arg0 : index) -> i32 {
  %1 = arith.index_castui %arg0 : index to i64
  %2 = arith.trunci %1 : i64 to i32
  // CHECK: %[[RESULT:.*]] = arith.index_castui %arg0 : index to i32
  // CHECH: util.return %[[RESULT]]
  util.return %2 : i32
}

// -----
// CHECK-LABEL: @elide_trunc_of_index_cast
util.func @elide_trunc_of_index_cast(%arg0 : index) -> i32 {
  %1 = arith.index_cast %arg0 : index to i64
  %2 = arith.trunci %1 : i64 to i32
  // CHECK: %[[RESULT:.*]] = arith.index_castui %arg0 : index to i32
  // CHECH: util.return %[[RESULT]]
  util.return %2 : i32
}

// -----
// CHECK-LABEL: @util_align_bounds_div
util.func @util_align_bounds_div(%arg0 : index, %arg1 : index) -> index, index, index, i1, i1 {
  %0 = util.assume.int %arg0<umin=10, umax=120> : index
  %1 = util.assume.int %arg1<umin=64, umax=64, udiv=64> : index
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64
  // CHECK-DAG: %[[ASSUME:.*]] = util.assume.int %arg0
  // CHECK: %[[ALIGN:.*]] = util.align %[[ASSUME]], %[[C64]]
  %2 = util.align %0, %1 : index

  // The result should be >= 64 and <= 128.
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %lower = arith.cmpi uge, %2, %c64 : index   // True
  %upper = arith.cmpi ule, %2, %c128 : index  // True
  %under = arith.cmpi ult, %2, %c64 : index   // False
  %over = arith.cmpi ugt, %2, %c128 : index   // False
  %in_bounds = arith.andi %lower, %upper : i1 // True
  %out_bounds = arith.andi %under, %over : i1 // False

  // And 64 should evenly divide it.
  %rem64 = arith.remui %2, %c64 : index
  // But 128 should not.
  // CHECK: %[[REM128:.*]] = arith.remui
  %rem128 = arith.remui %2, %c128 : index
  // CHECK: util.return %[[ALIGN]], %[[ZERO]], %[[REM128]], %[[TRUE]], %[[FALSE]]
  util.return %2, %rem64, %rem128, %in_bounds, %out_bounds : index, index, index, i1, i1
}

// -----
// Unbounded lhs of util.align technically has a range that extends to the max
// value of the bit width. Attempting to align this overflows (to zero). If not
// caught, this will most likely lead the optimizer to conclude that the
// aligned result is a constant zero. This code is verified by checking for
// overflow generally and should handle this case.
// CHECK-LABEL: @util_align_overflow
util.func @util_align_overflow(%arg0 : i64) -> i64 {
  %c64 = arith.constant 64 : i64
  // CHECK: util.align
  %0 = util.align %arg0, %c64 : i64
  util.return %0 : i64
}

// -----
// Aligning to an alignment of zero doesn't make a lot of sense but it isn't
// numerically an error. We don't fold or optimize this case and we verify
// it as such (and that other division by zero errors don't come up).
// CHECK-LABEL: @util_align_zero
util.func @util_align_zero(%arg0 : i64) -> i64 {
  %c0 = arith.constant 0 : i64
  %c16 = arith.constant 16 : i64
  %assume = util.assume.int %arg0<umin=0, umax=15> : i64
  %c128 = arith.constant 128 : i64
  // CHECK: util.align
  // CHECK: arith.remui
  %0 = util.align %assume, %c0 : i64
  %rem16 = arith.remui %0, %c16 : i64
  util.return %rem16 : i64
}

// -----

util.func @hal_buffer_view_dim_min_max(%bv : !hal.buffer_view) -> (i1, i1, i1) {
  %zero = arith.constant 0 : index
  %max = arith.constant 9007199254740991 : index
  %0 = hal.buffer_view.dim<%bv : !hal.buffer_view>[0] : index
  %1 = arith.cmpi slt, %0, %zero : index
  %2 = arith.cmpi uge, %0, %zero : index
  %3 = arith.cmpi ugt, %0, %max : index
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: util.return %[[FALSE]], %[[TRUE]], %[[FALSE]]
  util.return %1, %2, %3 : i1, i1, i1
}

// -----

util.func @hal_buffer_view_rank_min_max(%bv : !hal.buffer_view) -> (i1, i1, i1) {
  %zero = arith.constant 0 : index
  %max = arith.constant 4096 : index
  %0 = hal.buffer_view.rank<%bv : !hal.buffer_view> : index
  %1 = arith.cmpi slt, %0, %zero : index
  %2 = arith.cmpi uge, %0, %zero : index
  %3 = arith.cmpi ugt, %0, %max : index
  // CHECK-DAG: %[[FALSE:.*]] = arith.constant false
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK: util.return %[[FALSE]], %[[TRUE]], %[[FALSE]]
  util.return %1, %2, %3 : i1, i1, i1
}
