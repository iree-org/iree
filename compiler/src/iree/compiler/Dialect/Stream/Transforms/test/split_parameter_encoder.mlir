// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay' %s | FileCheck %s --check-prefix=OVERLAY
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay' %s | FileCheck %s --check-prefix=OVERLAY-MIXED
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=consolidate' %s | FileCheck %s --check-prefix=COMPARE-CONSOLIDATE
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay' %s | FileCheck %s --check-prefix=COMPARE-OVERLAY
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='output-scope=my_custom_scope' %s | FileCheck %s --check-prefix=SCOPE
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='max-encoding-growth-factor=2.0' %s | FileCheck %s --check-prefix=GROWTH2
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay' %s | FileCheck %s --check-prefix=EMPTY

// Tests simple constant with splat initialization.
// This is the most basic case - a global initialized with a constant splat.
// This should NOT be hoisted (no parameter input).

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @simple_constant : !stream.resource<constant>
util.global private @simple_constant : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  // CHECK: %[[C0_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[C1024:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[C0_I32]] : i32 -> !stream.resource<constant>{%[[C1024]]}
  %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1024}
  // CHECK: util.global.store %[[SPLAT]], @simple_constant : !stream.resource<constant>
  util.global.store %splat, @simple_constant : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests computed constant with transformation.
// This tests a constant that undergoes some computation (fill operation).
// Should NOT be hoisted (no parameter input).

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @computed_constant : !stream.resource<constant>
util.global private @computed_constant : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  // CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[C42_I32:.+]] = arith.constant 42 : i32
  %c42_i32 = arith.constant 42 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Create base splat.
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[C0_I32]] : i32 -> !stream.resource<constant>{%[[C1024]]}
  %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1024}

  // Fill a region with different value.
  // CHECK: %[[FILLED:.+]] = stream.async.fill %[[C42_I32]], %[[SPLAT]][%[[C0]] to %[[C256]] for %[[C256]]] : i32 -> %[[SPLAT]] as !stream.resource<constant>{%[[C1024]]}
  %filled = stream.async.fill %c42_i32, %splat[%c0 to %c256 for %c256] : i32 -> %splat as !stream.resource<constant>{%c1024}

  // CHECK: util.global.store %[[FILLED]], @computed_constant : !stream.resource<constant>
  util.global.store %filled, @computed_constant : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests multiple constants with different patterns.
// This tests that the pass can handle multiple globals, some with splat and some
// with more complex initialization.

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @constant_a : !stream.resource<constant>
util.global private @constant_a : !stream.resource<constant>
// CHECK: util.global private @constant_b : !stream.resource<constant>
util.global private @constant_b : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  // CHECK-DAG: %[[C0_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1_I32:.+]] = arith.constant 1 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index
  // CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // First constant: simple splat.
  // CHECK: %[[SPLAT_A:.+]] = stream.async.splat %[[C0_I32]] : i32 -> !stream.resource<constant>{%[[C512]]}
  %splat_a = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c512}
  // CHECK: util.global.store %[[SPLAT_A]], @constant_a : !stream.resource<constant>
  util.global.store %splat_a, @constant_a : !stream.resource<constant>

  // Second constant: different splat value and size.
  // CHECK: %[[SPLAT_B:.+]] = stream.async.splat %[[C1_I32]] : i32 -> !stream.resource<constant>{%[[C1024]]}
  %splat_b = stream.async.splat %c1_i32 : i32 -> !stream.resource<constant>{%c1024}
  // CHECK: util.global.store %[[SPLAT_B]], @constant_b : !stream.resource<constant>
  util.global.store %splat_b, @constant_b : !stream.resource<constant>

  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter transformation.
// This tests loading a parameter and applying a transformation (fill operation).
// This SHOULD be hoisted since it has a parameter input.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @parameter_transformed : !stream.resource<constant>
util.global private @parameter_transformed : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Load parameter from external source.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"param0"> : vector<1024xi8>

  // Fill a region with different value.
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] : i32 -> %param as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @parameter_transformed
  util.global.store %filled, @parameter_transformed : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests pure splat should NOT be hoisted (negative case).
// This tests that a pure splat with no inputs and no transformation is not hoisted.
// The pass should leave this module unchanged.

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @pure_splat_only : !stream.resource<constant>
util.global private @pure_splat_only : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  // CHECK: %[[C99_I32:.+]] = arith.constant 99 : i32
  %c99_i32 = arith.constant 99 : i32
  // CHECK: %[[C2048:.+]] = arith.constant 2048 : index
  %c2048 = arith.constant 2048 : index

  // Pure splat with no parameter input - should NOT be hoisted.
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[C99_I32]] : i32 -> !stream.resource<constant>{%[[C2048]]}
  %splat = stream.async.splat %c99_i32 : i32 -> !stream.resource<constant>{%c2048}

  // CHECK: util.global.store %[[SPLAT]], @pure_splat_only : !stream.resource<constant>
  util.global.store %splat, @pure_splat_only : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests a parameter transformed by a dispatch operation.
// Should be hoisted as it represents expensive computation on a parameter.
// Real-world: Elementwise operations, quantization, or encoding on weights.

stream.executable private @executable {
  stream.executable.export public @dispatch
}

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_with_dispatch : !stream.resource<constant>
util.global private @param_with_dispatch : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  // Load parameter from external source.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"dispatch_param"> : vector<1024xi8>

  // Dispatch performing operation on parameter.
  // CHECK-NOT: stream.async.dispatch
  %result = stream.async.dispatch @executable::@dispatch[%c1, %c1, %c1](%param[%c0 to %c1024 for %c1024]) :
    (!stream.resource<constant>{%c1024}) -> !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_with_dispatch : !stream.resource<constant>
  util.global.store %result, @param_with_dispatch : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter + splat + dispatch pattern.
// Splat should be cloned to consumers but not serialized (preferCloneToConsumers).
// Real-world: Parameter combined with constant baseline (e.g., weight + bias).

stream.executable private @executable {
  stream.executable.export public @add_dispatch
}

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_splat_dispatch : !stream.resource<constant>
util.global private @param_splat_dispatch : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  // Load parameter from external source.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"weights"> : vector<1024xi8>

  // Create splat (should be cloned but not serialized).
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: stream.async.splat
  %splat = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1024}

  // Dispatch using both parameter and splat.
  // CHECK-NOT: stream.async.dispatch
  %result = stream.async.dispatch @executable::@add_dispatch[%c1, %c1, %c1](
    %param[%c0 to %c1024 for %c1024],
    %splat[%c0 to %c1024 for %c1024]
  ) : (!stream.resource<constant>{%c1024}, !stream.resource<constant>{%c1024}) -> !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_splat_dispatch : !stream.resource<constant>
  util.global.store %result, @param_splat_dispatch : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter with metadata operations (subview).
// Metadata operations should not prevent hoisting.
// Real-world: Extract layer weights from combined parameter.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_metadata_ops : !stream.resource<constant>
util.global private @param_metadata_ops : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  // Load larger parameter.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"combined_param"> : vector<1024xi8>

  // Extract slice (metadata operation).
  // CHECK-NOT: stream.async.slice
  %slice = stream.async.slice %param[%c256 to %c512] : !stream.resource<constant>{%c1024} -> !stream.resource<constant>{%c256}

  // Apply transformation to slice.
  %c100_i32 = arith.constant 100 : i32
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c100_i32, %slice[%c0 to %c256 for %c256] : i32 -> %slice as !stream.resource<constant>{%c256}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_metadata_ops : !stream.resource<constant>
  util.global.store %filled, @param_metadata_ops : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests borderline growth (1.15x) - should pass.
// Within threshold growth should be allowed.
// Real-world: Small padding for alignment.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_acceptable_growth : !stream.resource<constant>
util.global private @param_acceptable_growth : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index
  %c1180 = arith.constant 1180 : index  // ~1.15x growth

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"param_growth"> : vector<1024xi8>

  // Slight growth for padding - should be within 1.2x threshold
  %c0 = arith.constant 0 : index
  %c156 = arith.constant 156 : index
  %c0_i32 = arith.constant 0 : i32

  // Create slightly larger buffer
  // CHECK-NOT: stream.async.splat
  %padded = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1180}

  // Copy parameter into padded buffer
  // CHECK-NOT: stream.async.update
  %result = stream.async.update %param, %padded[%c0 to %c1024] :
    !stream.resource<constant>{%c1024} -> %padded as !stream.resource<constant>{%c1180}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: %[[RESULT:.+]] = stream.resource.subview %[[PARAM]]
  // CHECK: util.global.store %[[RESULT]], @param_acceptable_growth : !stream.resource<constant>
  util.global.store %result, @param_acceptable_growth : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Control Flow in Initializers
//===----------------------------------------------------------------------===//

// Tests scf.for loop with fixed bounds.
// Loop should be unrolled if bounds are constant.
// Real-world: Fixed preprocessing iterations.
// Tests scf.for loop with constant bounds.
// Should hoist the loop and its body since bounds are constant.
// Real-world: Iterative parameter transformations.

// Encoder module should be generated with scf.for hoisted.
// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// Encoder should contain the scf.for loop with result scattered to parameter.
// CHECK: %[[IMPORT_TP:.+]] = stream.timepoint.import {{.+}} %arg1 : (!hal.fence) => !stream.timepoint
// CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK: %[[PACK_SIZE:.+]]:2 = stream.resource.pack {{.+}} slices({
// CHECK-NEXT:   [0, 0] = %[[C1024]]
// CHECK-NEXT: }) : index
// CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized {{.+}} await(%[[IMPORT_TP]]) => !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: %[[ALLOCA_READY:.+]] = stream.timepoint.await %[[ALLOCA_TP]] => %[[ALLOCA]] : !stream.resource<transient>{%[[PACK_SIZE]]#0}
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK: %[[PARAM_RESOURCE:.+]], %[[PARAM_TP:.+]] = stream.async.parameter.load {{.+}} await(%[[IMPORT_TP]]) "model"::"iterative_param"[%[[C0_I64]]] : !stream.resource<constant>{%[[C1024]]} => !stream.timepoint
// CHECK: %[[INPUT:.+]] = stream.timepoint.await %[[PARAM_TP]] => %[[PARAM_RESOURCE]] : !stream.resource<constant>{%[[C1024]]}
// CHECK: %[[LOOP_RESULT:.+]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG:.+]] = %[[INPUT]]) -> (!stream.resource<constant>) {
// CHECK:   %[[C100:.+]] = arith.constant 100 : i32
// CHECK:   %[[FILLED:.+]] = stream.async.fill %[[C100]], %[[ARG]][%[[C0]] to %[[C256]] for %[[C256]]] : i32 -> %[[ARG]] as !stream.resource<constant>{%[[C1024]]}
// CHECK:   scf.yield %[[FILLED]] : !stream.resource<constant>
// CHECK: }
// CHECK: %[[UPDATE_END:.+]] = arith.addi %[[PACK_SIZE]]#1, %[[C1024]] : index
// CHECK: %[[UPDATED:.+]] = stream.async.update {{.+}} %[[LOOP_RESULT]], %[[ALLOCA_READY]][%[[PACK_SIZE]]#1 to %[[UPDATE_END]]] : !stream.resource<constant>{%[[C1024]]} -> %[[ALLOCA_READY]] as !stream.resource<transient>{%[[PACK_SIZE]]#0}
// CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier {{.+}} %[[UPDATED]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: %[[SCATTER_RESULT:.+]], %[[SCATTER_TP:.+]] = stream.async.parameter.scatter {{.+}} await(%[[BARRIER_TP]]) {
// CHECK-NEXT:   %[[BARRIER_RESULT]][%[[PACK_SIZE]]#1 to %[[UPDATE_END]] for %[[C1024]]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} -> ""::"parameter0"[%[[C0_I64]]]
// CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
// CHECK: %[[JOIN_TP:.+]] = stream.timepoint.join max(%[[SCATTER_TP]]) => !stream.timepoint
// CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca {{.+}} await(%[[JOIN_TP]]) => %[[SCATTER_RESULT]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: stream.timepoint.chain_external {{.+}} %[[DEALLOCA_TP]] => (%arg2 : !hal.fence)

// Original module should have parameter load instead of scf.for.
// CHECK-LABEL: util.global private @scf_for_fixed_bounds
util.global private @scf_for_fixed_bounds : !stream.resource<constant>

util.initializer {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"iterative_param"> : vector<1024xi8>

  // Fixed-bound loop that could be unrolled.
  // CHECK-NOT: scf.for
  // CHECK-NOT: stream.async.fill
  %result = scf.for %i = %c0 to %c3 step %c1
    iter_args(%arg = %param) -> (!stream.resource<constant>) {
    // Apply transformation in each iteration.
    %c100_i32 = arith.constant 100 : i32
    %processed = stream.async.fill %c100_i32, %arg[%c0 to %c256 for %c256] :
      i32 -> %arg as !stream.resource<constant>{%c1024}
    scf.yield %processed : !stream.resource<constant>
  }

  // Original module loads from parameter instead of executing loop.
  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @scf_for_fixed_bounds
  util.global.store %result, @scf_for_fixed_bounds : !stream.resource<constant>
  util.return
}

// -----

// Tests scf.if conditional with compile-time constant condition.
// Should hoist the taken branch if condition is constant.
// Real-world: Conditional initialization for specific target.

// Encoder module should be generated with scf.if hoisted.
// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// Encoder should contain the scf.if conditional with result scattered to parameter.
// CHECK: %[[IMPORT_TP:.+]] = stream.timepoint.import {{.+}} %arg1 : (!hal.fence) => !stream.timepoint
// CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK: %[[PACK_SIZE:.+]]:2 = stream.resource.pack {{.+}} slices({
// CHECK-NEXT:   [0, 0] = %[[C1024]]
// CHECK-NEXT: }) : index
// CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized {{.+}} await(%[[IMPORT_TP]]) => !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: %[[ALLOCA_READY:.+]] = stream.timepoint.await %[[ALLOCA_TP]] => %[[ALLOCA]] : !stream.resource<transient>{%[[PACK_SIZE]]#0}
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK: %[[PARAM_RESOURCE:.+]], %[[PARAM_TP:.+]] = stream.async.parameter.load {{.+}} await(%[[IMPORT_TP]]) "model"::"conditional_param"[%[[C0_I64]]] : !stream.resource<constant>{%[[C1024]]} => !stream.timepoint
// CHECK: %[[INPUT:.+]] = stream.timepoint.await %[[PARAM_TP]] => %[[PARAM_RESOURCE]] : !stream.resource<constant>{%[[C1024]]}
// CHECK: %[[IF_RESULT:.+]] = scf.if %[[TRUE]] -> (!stream.resource<constant>) {
// CHECK:   %[[C42:.+]] = arith.constant 42 : i32
// CHECK:   %[[FILLED:.+]] = stream.async.fill %[[C42]], %[[INPUT]][%[[C0]] to %[[C256]] for %[[C256]]] : i32 -> %[[INPUT]] as !stream.resource<constant>{%[[C1024]]}
// CHECK:   scf.yield %[[FILLED]] : !stream.resource<constant>
// CHECK: } else {
// CHECK:   scf.yield %[[INPUT]] : !stream.resource<constant>
// CHECK: }
// CHECK: %[[UPDATE_END:.+]] = arith.addi %[[PACK_SIZE]]#1, %[[C1024]] : index
// CHECK: %[[UPDATED:.+]] = stream.async.update {{.+}} %[[IF_RESULT]], %[[ALLOCA_READY]][%[[PACK_SIZE]]#1 to %[[UPDATE_END]]] : !stream.resource<constant>{%[[C1024]]} -> %[[ALLOCA_READY]] as !stream.resource<transient>{%[[PACK_SIZE]]#0}
// CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier {{.+}} %[[UPDATED]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: %[[SCATTER_RESULT:.+]], %[[SCATTER_TP:.+]] = stream.async.parameter.scatter {{.+}} await(%[[BARRIER_TP]]) {
// CHECK-NEXT:   %[[BARRIER_RESULT]][%[[PACK_SIZE]]#1 to %[[UPDATE_END]] for %[[C1024]]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} -> ""::"parameter0"[%[[C0_I64]]]
// CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
// CHECK: %[[JOIN_TP:.+]] = stream.timepoint.join max(%[[SCATTER_TP]]) => !stream.timepoint
// CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca {{.+}} await(%[[JOIN_TP]]) => %[[SCATTER_RESULT]] : !stream.resource<transient>{%[[PACK_SIZE]]#0} => !stream.timepoint
// CHECK: stream.timepoint.chain_external {{.+}} %[[DEALLOCA_TP]] => (%arg2 : !hal.fence)

// Original module should have parameter load instead of scf.if.
// CHECK-LABEL: util.global private @scf_if_constant_condition
util.global private @scf_if_constant_condition : !stream.resource<constant>

util.initializer {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"conditional_param"> : vector<1024xi8>

  // Conditional with compile-time constant.
  // CHECK-NOT: scf.if
  // CHECK-NOT: stream.async.fill
  %result = scf.if %true -> (!stream.resource<constant>) {
    // True branch - should be taken.
    %c42_i32 = arith.constant 42 : i32
    %processed = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] :
      i32 -> %param as !stream.resource<constant>{%c1024}
    scf.yield %processed : !stream.resource<constant>
  } else {
    // False branch - should be eliminated.
    scf.yield %param : !stream.resource<constant>
  }

  // Original module loads from parameter instead of executing conditional.
  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @scf_if_constant_condition
  util.global.store %result, @scf_if_constant_condition : !stream.resource<constant>
  util.return
}

// -----

//===----------------------------------------------------------------------===//
// Multiple Outputs from Single Parameter
//===----------------------------------------------------------------------===//

// Tests single parameter producing multiple transformed outputs.
// Should hoist both transformations, outputs packed.
// Real-world: Different quantization formats for different layers.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK-DAG: util.global private @single_param_multi_output_a : !stream.resource<constant>
util.global private @single_param_multi_output_a : !stream.resource<constant>
// CHECK-DAG: util.global private @single_param_multi_output_b : !stream.resource<constant>
util.global private @single_param_multi_output_b : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c768 = arith.constant 768 : index
  %c1024 = arith.constant 1024 : index

  // Single parameter input
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"shared_param"> : vector<1024xi8>

  // First transformation
  %c100_i32 = arith.constant 100 : i32
  // CHECK-NOT: stream.async.fill
  %output_a = stream.async.fill %c100_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // Second transformation
  %c200_i32 = arith.constant 200 : i32
  %output_b = stream.async.fill %c200_i32, %param[%c512 to %c768 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // Both outputs are packed into a single parameter, loaded twice and extracted via subviews.
  // CHECK-DAG: %[[PACKED_SIZE:.+]] = arith.constant 2048 : index
  // CHECK-DAG: %[[SUBVIEW_OFFSET_0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[SUBVIEW_SIZE:.+]] = arith.constant 1024 : index
  // CHECK-DAG: %[[PARAM_A:.+]] = stream.async.constant : !stream.resource<constant>{%[[PACKED_SIZE]]} = #stream.parameter.named<""::"parameter0">
  // CHECK-DAG: %[[RESULT_A:.+]] = stream.resource.subview %[[PARAM_A]][%[[SUBVIEW_OFFSET_0]]] : !stream.resource<constant>{%[[PACKED_SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  // CHECK-DAG: util.global.store %[[RESULT_A]], @single_param_multi_output_a : !stream.resource<constant>
  util.global.store %output_a, @single_param_multi_output_a : !stream.resource<constant>
  // CHECK-DAG: %[[PARAM_B:.+]] = stream.async.constant : !stream.resource<constant>{%[[PACKED_SIZE]]} = #stream.parameter.named<""::"parameter0">
  // CHECK-DAG: %[[RESULT_B:.+]] = stream.resource.subview %[[PARAM_B]][%[[SUBVIEW_SIZE]]] : !stream.resource<constant>{%[[PACKED_SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  // CHECK-DAG: util.global.store %[[RESULT_B]], @single_param_multi_output_b : !stream.resource<constant>
  util.global.store %output_b, @single_param_multi_output_b : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Device Specialization & Affinity
//===----------------------------------------------------------------------===//

// Tests parameter with affinity annotation.
// Should hoist with affinity preserved in encoder.
// Real-world: GPU-specific parameter transformation.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_with_affinity : !stream.resource<constant>
util.global private @param_with_affinity : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Parameter with device affinity
  %param = stream.async.constant on(#hal.device.affinity<@device_0>) :
    !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"gpu_param"> : vector<1024xi8>

  // Transformation maintaining affinity
  %c42_i32 = arith.constant 42 : i32
  // CHECK-NOT: stream.async.fill
  %result = stream.async.fill on(#hal.device.affinity<@device_0>) %c42_i32,
    %param[%c0 to %c256 for %c256] : i32 -> %param as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_with_affinity : !stream.resource<constant>
  util.global.store %result, @param_with_affinity : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}


// -----

//===----------------------------------------------------------------------===//
// Stress Tests
//===----------------------------------------------------------------------===//

// Tests very small parameter (1 byte).
// Should handle minimum size parameters.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @minimum_size_param : !stream.resource<constant>
util.global private @minimum_size_param : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1 = arith.constant 1 : index

  // Tiny 1-byte parameter
  %param = stream.async.constant : !stream.resource<constant>{%c1} =
    #stream.parameter.named<"model"::"tiny"> : vector<1xi8>

  // Even tiny transform should work
  %c0 = arith.constant 0 : index
  %c42_i32 = arith.constant 42 : i32
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param[%c0 to %c1 for %c1] :
    i32 -> %param as !stream.resource<constant>{%c1}

  // CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[PARAM:.+]] = stream.async.constant : !stream.resource<constant>{%[[C64]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[RESULT:.+]] = stream.resource.subview %[[PARAM]][%[[C0]]] : !stream.resource<constant>{%[[C64]]} -> !stream.resource<constant>{%[[C1]]}
  // CHECK: util.global.store %[[RESULT]], @minimum_size_param : !stream.resource<constant>
  util.global.store %filled, @minimum_size_param : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests deep expression DAG (multiple levels of operations).
// Should handle deep computation chains.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @deep_expression_dag : !stream.resource<constant>
util.global private @deep_expression_dag : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c192 = arith.constant 192 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Load parameter
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"deep_param"> : vector<1024xi8>

  // Deep chain of transformations
  // CHECK-NOT: stream.async.fill
  %c1_i32 = arith.constant 1 : i32
  %stage1 = stream.async.fill %c1_i32, %param[%c0 to %c64 for %c64] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  %c2_i32 = arith.constant 2 : i32
  %stage2 = stream.async.fill %c2_i32, %stage1[%c64 to %c128 for %c64] :
    i32 -> %stage1 as !stream.resource<constant>{%c1024}

  %c3_i32 = arith.constant 3 : i32
  %stage3 = stream.async.fill %c3_i32, %stage2[%c128 to %c192 for %c64] :
    i32 -> %stage2 as !stream.resource<constant>{%c1024}

  %c4_i32 = arith.constant 4 : i32
  %stage4 = stream.async.fill %c4_i32, %stage3[%c192 to %c256 for %c64] :
    i32 -> %stage3 as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @deep_expression_dag : !stream.resource<constant>
  util.global.store %stage4, @deep_expression_dag : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Advanced Growth Factor Tests
//===----------------------------------------------------------------------===//

// Tests exact 1.2x growth threshold - should pass.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @exact_growth_threshold : !stream.resource<constant>
util.global private @exact_growth_threshold : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1000 = arith.constant 1000 : index
  %c1200 = arith.constant 1200 : index  // Exactly 1.2x

  %param = stream.async.constant : !stream.resource<constant>{%c1000} =
    #stream.parameter.named<"model"::"exact_threshold"> : vector<1000xi8>

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: stream.async.splat
  %padded = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1200}
  // CHECK-NOT: stream.async.update
  %result = stream.async.update %param, %padded[%c0 to %c1000] :
    !stream.resource<constant>{%c1000} -> %padded as !stream.resource<constant>{%c1200}

  // CHECK-DAG: %[[C1216:.+]] = arith.constant 1216 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1200:.+]] = arith.constant 1200 : index
  // CHECK: %[[PARAM:.+]] = stream.async.constant : !stream.resource<constant>{%[[C1216]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[RESULT:.+]] = stream.resource.subview %[[PARAM]][%[[C0]]] : !stream.resource<constant>{%[[C1216]]} -> !stream.resource<constant>{%[[C1200]]}
  // CHECK: util.global.store %[[RESULT]], @exact_growth_threshold : !stream.resource<constant>
  util.global.store %result, @exact_growth_threshold : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests just over 1.2x growth (1.21x) - should reject.

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @over_growth_threshold : !stream.resource<constant>
util.global private @over_growth_threshold : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  // CHECK-DAG: %[[C1000:.+]] = arith.constant 1000 : index
  %c1000 = arith.constant 1000 : index
  // CHECK-DAG: %[[C1210:.+]] = arith.constant 1210 : index
  %c1210 = arith.constant 1210 : index  // 1.21x - over threshold

  // CHECK: %[[PARAM:.+]] = stream.async.constant : !stream.resource<constant>{%[[C1000]]} = #stream.parameter.named<"model"::"over_threshold">
  %param = stream.async.constant : !stream.resource<constant>{%c1000} =
    #stream.parameter.named<"model"::"over_threshold"> : vector<1000xi8>

  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[PADDED:.+]] = stream.async.splat
  %padded = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1210}
  // CHECK: %[[RESULT:.+]] = stream.async.update %[[PARAM]], %[[PADDED]]
  %result = stream.async.update %param, %padded[%c0 to %c1000] :
    !stream.resource<constant>{%c1000} -> %padded as !stream.resource<constant>{%c1210}

  // CHECK: util.global.store %[[RESULT]], @over_growth_threshold : !stream.resource<constant>
  util.global.store %result, @over_growth_threshold : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Complex Data Flow Patterns
//===----------------------------------------------------------------------===//

// Tests parameter used by multiple operations (wide DAG).
// Single parameter with many consumers.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK-DAG: util.global private @wide_expression_dag_a : !stream.resource<constant>
util.global private @wide_expression_dag_a : !stream.resource<constant>
// CHECK-DAG: util.global private @wide_expression_dag_b : !stream.resource<constant>
util.global private @wide_expression_dag_b : !stream.resource<constant>
// CHECK-DAG: util.global private @wide_expression_dag_c : !stream.resource<constant>
util.global private @wide_expression_dag_c : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %c1024 = arith.constant 1024 : index

  // Single parameter used by many operations
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"wide_param"> : vector<1024xi8>

  // Many transformations using the same parameter
  %c10_i32 = arith.constant 10 : i32
  // CHECK-NOT: stream.async.fill
  %out_a = stream.async.fill %c10_i32, %param[%c0 to %c100 for %c100] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  %c20_i32 = arith.constant 20 : i32
  %out_b = stream.async.fill %c20_i32, %param[%c100 to %c200 for %c100] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  %c30_i32 = arith.constant 30 : i32
  %out_c = stream.async.fill %c30_i32, %param[%c200 to %c300 for %c100] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // All outputs packed into a single parameter and extracted via subviews.
  // CHECK-DAG: %[[PACKED_SIZE:.+]] = arith.constant 3072 : index
  // CHECK-DAG: %[[OFFSET_0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[SUBVIEW_SIZE:.+]] = arith.constant 1024 : index
  // CHECK-DAG: %[[OFFSET_2048:.+]] = arith.constant 2048 : index
  // CHECK-DAG: %[[PARAM_A:.+]] = stream.async.constant : !stream.resource<constant>{%[[PACKED_SIZE]]} = #stream.parameter.named<""::"parameter0">
  // CHECK-DAG: %[[RESULT_A:.+]] = stream.resource.subview %[[PARAM_A]][%[[OFFSET_0]]] : !stream.resource<constant>{%[[PACKED_SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  // CHECK-DAG: util.global.store %[[RESULT_A]], @wide_expression_dag_a : !stream.resource<constant>
  util.global.store %out_a, @wide_expression_dag_a : !stream.resource<constant>
  // CHECK-DAG: %[[PARAM_B:.+]] = stream.async.constant : !stream.resource<constant>{%[[PACKED_SIZE]]} = #stream.parameter.named<""::"parameter0">
  // CHECK-DAG: %[[RESULT_B:.+]] = stream.resource.subview %[[PARAM_B]][%[[SUBVIEW_SIZE]]] : !stream.resource<constant>{%[[PACKED_SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  // CHECK-DAG: util.global.store %[[RESULT_B]], @wide_expression_dag_b : !stream.resource<constant>
  util.global.store %out_b, @wide_expression_dag_b : !stream.resource<constant>
  // CHECK-DAG: %[[PARAM_C:.+]] = stream.async.constant : !stream.resource<constant>{%[[PACKED_SIZE]]} = #stream.parameter.named<""::"parameter0">
  // CHECK-DAG: %[[RESULT_C:.+]] = stream.resource.subview %[[PARAM_C]][%[[OFFSET_2048]]] : !stream.resource<constant>{%[[PACKED_SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  // CHECK-DAG: util.global.store %[[RESULT_C]], @wide_expression_dag_c : !stream.resource<constant>
  util.global.store %out_c, @wide_expression_dag_c : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter transformation with clone operation.
// Clone operations should be handled (may have preferCloneToConsumers).

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_with_clone : !stream.resource<constant>
util.global private @param_with_clone : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"clone_param"> : vector<1024xi8>

  // Clone operation (might have preferCloneToConsumers)
  // CHECK-NOT: stream.async.clone
  %cloned = stream.async.clone %param : !stream.resource<constant>{%c1024} ->
    !stream.resource<constant>{%c1024}

  // Transform the clone
  %c99_i32 = arith.constant 99 : i32
  // CHECK-NOT: stream.async.fill
  %result = stream.async.fill %c99_i32, %cloned[%c0 to %c256 for %c256] :
    i32 -> %cloned as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_with_clone : !stream.resource<constant>
  util.global.store %result, @param_with_clone : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter transformation with clone at END of expression.
// This tests findProducedValue skipping past final clone to find producer.
// Pattern: param → clone(to *) → dispatch → clone(to constant) → store.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

stream.executable private @dispatch_for_clone_test {
  stream.executable.export public @fill
}

// CHECK-LABEL: module {
// CHECK: util.global private @param_with_trailing_clone : !stream.resource<constant>
util.global private @param_with_trailing_clone : !stream.resource<constant>

// The original ops (clone → dispatch → clone) should all be hoisted to encoder.
// CHECK:       util.initializer {
// CHECK-NOT:     stream.async.clone
// CHECK-NOT:     stream.async.dispatch
// CHECK:         %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
// CHECK:         util.global.store %[[PARAM]], @param_with_trailing_clone
// CHECK:         util.return
// CHECK:       }
util.initializer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"trailing_clone_param"> : vector<1024xi8>

  // Clone to unknown lifetime for dispatch input.
  %for_dispatch = stream.async.clone %param :
    !stream.resource<constant>{%c1024} -> !stream.resource<*>{%c1024}

  // Dispatch transforms the parameter.
  %dispatched = stream.async.dispatch @dispatch_for_clone_test::@fill[%c1, %c1, %c1](%for_dispatch[%c0 to %c1024 for %c1024]) :
    (!stream.resource<*>{%c1024}) -> !stream.resource<*>{%c1024}

  // Clone at END of expression back to constant lifetime.
  // findProducedValue must skip this to find the dispatch as the producer.
  %result = stream.async.clone %dispatched :
    !stream.resource<*>{%c1024} -> !stream.resource<constant>{%c1024}

  util.global.store %result, @param_with_trailing_clone : !stream.resource<constant>
  util.return
}

// -----

//===----------------------------------------------------------------------===//
// Transfer Operations
//===----------------------------------------------------------------------===//

// Tests parameter with transfer operations.
// Transfers should be handled correctly.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_transfer : !stream.resource<constant>
util.global private @param_transfer : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"transfer_param"> : vector<1024xi8>

  // Transfer to different lifetime (if needed)
  // CHECK-NOT: stream.async.transfer
  %transferred = stream.async.transfer %param :
    !stream.resource<constant>{%c1024} -> !stream.resource<constant>{%c1024}

  // Transform transferred value
  %c88_i32 = arith.constant 88 : i32
  // CHECK-NOT: stream.async.fill
  %result = stream.async.fill %c88_i32, %transferred[%c0 to %c256 for %c256] :
    i32 -> %transferred as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_transfer : !stream.resource<constant>
  util.global.store %result, @param_transfer : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests parameter copying (combining two parameters).
// Copy operations should be hoisted to encoder.
// Real-world: Combining parameter shards into single buffer.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK: util.global private @param_copy_combine : !stream.resource<constant>
util.global private @param_copy_combine : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  // Load two parameters that will be combined.
  %param1 = stream.async.constant : !stream.resource<constant>{%c512} =
    #stream.parameter.named<"model"::"shard0"> : vector<512xi8>
  %param2 = stream.async.constant : !stream.resource<constant>{%c512} =
    #stream.parameter.named<"model"::"shard1"> : vector<512xi8>

  // Create destination buffer.
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NOT: stream.async.splat
  %combined = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c1024}

  // Copy first parameter.
  // CHECK-NOT: stream.async.copy
  %with_first = stream.async.copy %param1[%c0 to %c512], %combined[%c0 to %c512], %c512 :
    !stream.resource<constant>{%c512} -> %combined as !stream.resource<constant>{%c1024}

  // Copy second parameter.
  // CHECK-NOT: stream.async.copy
  %result = stream.async.copy %param2[%c0 to %c512], %with_first[%c512 to %c1024], %c512 :
    !stream.resource<constant>{%c512} -> %with_first as !stream.resource<constant>{%c1024}

  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_copy_combine : !stream.resource<constant>
  util.global.store %result, @param_copy_combine : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Multiple Initializers
//===----------------------------------------------------------------------===//

// Tests multiple initializers in same module.
// All should be processed independently.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all
// CHECK: util.func public @__encode_parameters_all

// CHECK-LABEL: module {
// CHECK-DAG: util.global private @multi_init_a : !stream.resource<constant>
util.global private @multi_init_a : !stream.resource<constant>
// CHECK-DAG: util.global private @multi_init_b : !stream.resource<constant>
util.global private @multi_init_b : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c512 = arith.constant 512 : index
  %param_a = stream.async.constant : !stream.resource<constant>{%c512} =
    #stream.parameter.named<"model"::"init_a"> : vector<512xi8>

  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c11_i32 = arith.constant 11 : i32
  // CHECK-NOT: stream.async.fill
  %result_a = stream.async.fill %c11_i32, %param_a[%c0 to %c256 for %c256] :
    i32 -> %param_a as !stream.resource<constant>{%c512}

  // CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
  // CHECK: %[[PARAM_A:.+]] = stream.async.constant : !stream.resource<constant>{%[[C1024]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[RESULT_A:.+]] = stream.resource.subview %[[PARAM_A]][%[[C0]]] : !stream.resource<constant>{%[[C1024]]} -> !stream.resource<constant>{%[[C512]]}
  // CHECK: util.global.store %[[RESULT_A]], @multi_init_a : !stream.resource<constant>
  util.global.store %result_a, @multi_init_a : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// CHECK: util.initializer {
util.initializer {
  %c512 = arith.constant 512 : index
  %param_b = stream.async.constant : !stream.resource<constant>{%c512} =
    #stream.parameter.named<"model"::"init_b"> : vector<512xi8>

  %c256 = arith.constant 256 : index
  %c512_0 = arith.constant 512 : index
  %c22_i32 = arith.constant 22 : i32
  // CHECK-NOT: stream.async.fill
  %result_b = stream.async.fill %c22_i32, %param_b[%c256 to %c512_0 for %c256] :
    i32 -> %param_b as !stream.resource<constant>{%c512}

  // CHECK-DAG: %[[C1024_0:.+]] = arith.constant 1024 : index
  // CHECK-DAG: %[[C512_0:.+]] = arith.constant 512 : index
  // CHECK: %[[PARAM_B:.+]] = stream.async.constant : !stream.resource<constant>{%[[C1024_0]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[RESULT_B:.+]] = stream.resource.subview %[[PARAM_B]][%[[C512_0]]] : !stream.resource<constant>{%[[C1024_0]]} -> !stream.resource<constant>{%[[C512_0]]}
  // CHECK: util.global.store %[[RESULT_B]], @multi_init_b : !stream.resource<constant>
  util.global.store %result_b, @multi_init_b : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Resource Lifetime Tests
//===----------------------------------------------------------------------===//

// Tests non-constant resource lifetime (should skip).

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
// CHECK: util.global private @non_constant_lifetime : !stream.resource<transient>
util.global private @non_constant_lifetime : !stream.resource<transient>

// CHECK: util.initializer {
util.initializer {
  // CHECK: %[[C1024:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Transient resource (not constant) - should skip
  // CHECK: %[[C0_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[TRANSIENT:.+]] = stream.async.splat %[[C0_I32]] : i32 -> !stream.resource<transient>{%[[C1024]]}
  %transient = stream.async.splat %c0_i32 : i32 -> !stream.resource<transient>{%c1024}

  // CHECK: util.global.store %[[TRANSIENT]], @non_constant_lifetime : !stream.resource<transient>
  util.global.store %transient, @non_constant_lifetime : !stream.resource<transient>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

//===----------------------------------------------------------------------===//
// Mode Testing: Consolidate vs Overlay
//===----------------------------------------------------------------------===//

// Tests pass-through parameter in consolidate mode (default).
// A parameter loaded and stored directly with no transformation should be
// included in the encoder output when in consolidate mode.

// CHECK: module @encoder
// CHECK: util.func public @__encode_parameter_detect_target
// CHECK: util.func public @__encode_parameter_indices_all
// CHECK: util.func public @__encode_parameter_steps_all

// CHECK-LABEL: module {
// CHECK: util.global private @passthrough_consolidate : !stream.resource<constant>
util.global private @passthrough_consolidate : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index

  // Load parameter directly without transformation.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"passthrough_param"> : vector<1024xi8>

  // Store directly - this is a pass-through (no transformation).
  // In consolidate mode, this should be included in encoder output.
  // CHECK: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @passthrough_consolidate
  util.global.store %param, @passthrough_consolidate : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests pass-through parameter in overlay mode.
// Same as previous test but with overlay mode enabled.
// The parameter should NOT be included in the encoder output since it's
// unmodified (includeUnmodified=false in overlay mode).

// Anchor to this specific test's main module
// OVERLAY-LABEL: util.global private @passthrough_overlay : !stream.resource<constant>
util.global private @passthrough_overlay : !stream.resource<constant>

// OVERLAY: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index

  // Load parameter directly without transformation.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"passthrough_param_overlay"> : vector<1024xi8>

  // Store directly - pass-through with no transformation.
  // In overlay mode, this should NOT be in encoder output.
  // The original parameter load should remain unchanged.
  // OVERLAY: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<"model"::"passthrough_param_overlay">
  // OVERLAY: util.global.store %[[PARAM]], @passthrough_overlay
  util.global.store %param, @passthrough_overlay : !stream.resource<constant>
  // OVERLAY: util.return
  util.return
  // OVERLAY: }
}

// -----

// Tests mixed parameters in consolidate mode.
// One parameter with transformation, one pass-through.
// Consolidate mode should include both in encoder output.

// CHECK-LABEL: util.global private @mixed_transformed : !stream.resource<constant>
util.global private @mixed_transformed : !stream.resource<constant>
// CHECK: util.global private @mixed_passthrough : !stream.resource<constant>
util.global private @mixed_passthrough : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Parameter 1: Transformed with fill operation.
  %param1 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"mixed_param1"> : vector<1024xi8>
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param1[%c0 to %c256 for %c256] : i32 -> %param1 as !stream.resource<constant>{%c1024}

  // Parameter 2: Pass-through (no transformation).
  %param2 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"mixed_param2"> : vector<1024xi8>

  // In consolidate mode, both should be loaded from encoder output.
  // CHECK-DAG: %[[C2048:.+]] = arith.constant 2048 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
  // CHECK: %[[PARAM:.+]] = stream.async.constant : !stream.resource<constant>{%[[C2048]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[SUBVIEW1:.+]] = stream.resource.subview %[[PARAM]][%[[C0]]] : !stream.resource<constant>{%[[C2048]]} -> !stream.resource<constant>{%[[C1024]]}
  // CHECK: util.global.store %[[SUBVIEW1]], @mixed_transformed
  util.global.store %filled, @mixed_transformed : !stream.resource<constant>

  // CHECK: %[[PARAM_0:.+]] = stream.async.constant : !stream.resource<constant>{%[[C2048]]} = #stream.parameter.named<""::"parameter0">
  // CHECK: %[[SUBVIEW2:.+]] = stream.resource.subview %[[PARAM_0]][%[[C1024]]] : !stream.resource<constant>{%[[C2048]]} -> !stream.resource<constant>{%[[C1024]]}
  // CHECK: util.global.store %[[SUBVIEW2]], @mixed_passthrough
  util.global.store %param2, @mixed_passthrough : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests mixed parameters in overlay mode.
// One parameter with transformation, one pass-through.
// Overlay mode should only include the transformed parameter.

// Anchor to the main module's first global to scope checks to this section
// OVERLAY-MIXED-LABEL: util.global private @mixed_transformed_overlay : !stream.resource<constant>
util.global private @mixed_transformed_overlay : !stream.resource<constant>
// OVERLAY-MIXED: util.global private @mixed_passthrough_overlay : !stream.resource<constant>
util.global private @mixed_passthrough_overlay : !stream.resource<constant>

// OVERLAY-MIXED: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Parameter 1: Transformed with fill operation.
  %param1 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"mixed_param1_overlay"> : vector<1024xi8>
  // OVERLAY-MIXED-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param1[%c0 to %c256 for %c256] : i32 -> %param1 as !stream.resource<constant>{%c1024}

  // Parameter 2: Pass-through (no transformation).
  %param2 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"mixed_param2_overlay"> : vector<1024xi8>

  // Overlay mode: transformed parameter from encoder, pass-through from original.
  // Parameters can be loaded in any order (SSA), use DAG to allow flexibility.
  // OVERLAY-MIXED-DAG: %{{.+}} = stream.async.constant {{.+}} #stream.parameter.named<"model"::"mixed_param2_overlay">
  // OVERLAY-MIXED-DAG: %{{.+}} = stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">

  // Stores should happen in this order.
  // OVERLAY-MIXED: util.global.store %{{.+}}, @mixed_transformed_overlay
  util.global.store %filled, @mixed_transformed_overlay : !stream.resource<constant>

  // OVERLAY-MIXED: util.global.store %{{.+}}, @mixed_passthrough_overlay
  util.global.store %param2, @mixed_passthrough_overlay : !stream.resource<constant>
  // OVERLAY-MIXED: util.return
  util.return
  // OVERLAY-MIXED: }
}

// -----

// Tests side-by-side mode comparison.
// Same input tested with both consolidate and overlay modes using different
// check prefixes to verify behavioral differences.

// Anchor to this test's unique globals.
// COMPARE-CONSOLIDATE-LABEL: util.global private @compare_transformed : !stream.resource<constant>
// COMPARE-OVERLAY-LABEL: util.global private @compare_transformed : !stream.resource<constant>
util.global private @compare_transformed : !stream.resource<constant>
// COMPARE-CONSOLIDATE: util.global private @compare_passthrough : !stream.resource<constant>
// COMPARE-OVERLAY: util.global private @compare_passthrough : !stream.resource<constant>
util.global private @compare_passthrough : !stream.resource<constant>

// COMPARE-CONSOLIDATE: util.initializer {
// COMPARE-OVERLAY: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Transformed parameter.
  %param1 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"compare_param1"> : vector<1024xi8>
  // COMPARE-CONSOLIDATE-NOT: stream.async.fill
  // COMPARE-OVERLAY-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param1[%c0 to %c256 for %c256] : i32 -> %param1 as !stream.resource<constant>{%c1024}

  // Pass-through parameter.
  %param2 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"compare_param2"> : vector<1024xi8>

  // Consolidate: Both from encoder output, packed into single parameter0, then subviewed.
  // Just verify key operations exist without strict ordering.
  // COMPARE-CONSOLIDATE-DAG: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // COMPARE-CONSOLIDATE-DAG: stream.resource.subview
  // COMPARE-CONSOLIDATE-DAG: util.global.store %{{.+}}, @compare_transformed
  util.global.store %filled, @compare_transformed : !stream.resource<constant>

  // COMPARE-CONSOLIDATE-DAG: util.global.store %{{.+}}, @compare_passthrough

  // Overlay: Transformed from encoder (parameter0), pass-through from original.
  // COMPARE-OVERLAY-DAG: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // COMPARE-OVERLAY-DAG: stream.async.constant {{.+}} #stream.parameter.named<"model"::"compare_param2">
  // COMPARE-OVERLAY-DAG: util.global.store %{{.+}}, @compare_transformed

  // COMPARE-OVERLAY-DAG: util.global.store %{{.+}}, @compare_passthrough
  util.global.store %param2, @compare_passthrough : !stream.resource<constant>

  // COMPARE-CONSOLIDATE: util.return
  // COMPARE-OVERLAY: util.return
  util.return
  // COMPARE-CONSOLIDATE: }
  // COMPARE-OVERLAY: }
}

// -----

// Tests custom output scope.
// Verifies that the encoder uses a custom scope name instead of default "encoded".

// Anchor to this test's unique global.
// SCOPE-LABEL: util.global private @custom_scope_global : !stream.resource<constant>
util.global private @custom_scope_global : !stream.resource<constant>

// SCOPE: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index

  // Parameter with transformation.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"custom_scope_param"> : vector<1024xi8>
  // SCOPE-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] : i32 -> %param as !stream.resource<constant>{%c1024}

  // Should load from custom scope "my_custom_scope" instead of default "encoded".
  // SCOPE: %[[PARAM:.+]] = stream.async.constant {{.+}} #stream.parameter.named<"my_custom_scope"::"parameter0">
  // SCOPE: util.global.store %[[PARAM]], @custom_scope_global
  util.global.store %filled, @custom_scope_global : !stream.resource<constant>
  // SCOPE: util.return
  util.return
  // SCOPE: }
}

// -----

// Tests growth factor threshold with increased limit.
// A parameter that grows 1.8x should be rejected with default threshold (1.2x)
// but accepted with custom threshold (2.0x).

// Anchor to this test's unique global.
// GROWTH2-LABEL: util.global private @growth_factor_test : !stream.resource<constant>
util.global private @growth_factor_test : !stream.resource<constant>

// GROWTH2: util.initializer {
util.initializer {
  %c42_i32 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c1800 = arith.constant 1800 : index

  // Parameter that grows from 1000 bytes (input) to 1800 bytes (after fill/pad).
  // 1.8x growth exceeds default 1.2x threshold but passes with 2.0x threshold.
  %param = stream.async.constant : !stream.resource<constant>{%c1000} =
    #stream.parameter.named<"model"::"growth_param"> : vector<1000xi8>

  // Fill operation that expands the parameter size (1000 -> 1800 bytes).
  // GROWTH2-NOT: stream.async.fill
  %expanded = stream.async.fill %c42_i32, %param[%c0 to %c1800 for %c1800] : i32 -> %param as !stream.resource<constant>{%c1800}

  // With growth factor 2.0, this should be hoisted (1.8x < 2.0).
  // Verify transformation was hoisted: parameter loads from encoder output.
  // GROWTH2-DAG: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // GROWTH2-DAG: util.global.store %{{.+}}, @growth_factor_test
  util.global.store %expanded, @growth_factor_test : !stream.resource<constant>
  // GROWTH2: util.return
  util.return
  // GROWTH2: }
}

// -----

// Tests empty encoder module in overlay mode.
// When all parameters are pass-through (no transformations) and in overlay mode,
// no encoder module should be generated since there's nothing to encode.

// Anchor to this test's unique global.
// EMPTY-LABEL: util.global private @empty_test_1 : !stream.resource<constant>
util.global private @empty_test_1 : !stream.resource<constant>
// EMPTY: util.global private @empty_test_2 : !stream.resource<constant>
util.global private @empty_test_2 : !stream.resource<constant>

// EMPTY: util.initializer {
util.initializer {
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  // Pass-through parameter 1 (no transformation).
  %param1 = stream.async.constant : !stream.resource<constant>{%c512} =
    #stream.parameter.named<"model"::"empty_param1"> : vector<512xi8>

  // Pass-through parameter 2 (no transformation).
  %param2 = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"empty_param2"> : vector<1024xi8>

  // Both should load from original parameters (no encoder output).
  // EMPTY-DAG: stream.async.constant {{.+}} #stream.parameter.named<"model"::"empty_param1">
  // EMPTY-DAG: stream.async.constant {{.+}} #stream.parameter.named<"model"::"empty_param2">
  // EMPTY-DAG: util.global.store %{{.+}}, @empty_test_1
  // EMPTY-DAG: util.global.store %{{.+}}, @empty_test_2
  util.global.store %param1, @empty_test_1 : !stream.resource<constant>

  util.global.store %param2, @empty_test_2 : !stream.resource<constant>

  // EMPTY: util.return
  util.return
  // EMPTY: }
}

// -----

//===----------------------------------------------------------------------===//
// Multi-Block Slice Ordering Tests
//===----------------------------------------------------------------------===//
// These tests exercise the slice ordering logic when operations span multiple
// blocks or regions. The backward slice collection must maintain proper
// topological order even when captured values from nested regions are involved.

// Tests that captured values from scf.if regions are handled correctly.
// This exercises the multi-root slice ordering logic where values defined
// outside an scf.if are used inside its regions.

stream.executable private @captured_dispatch {
  stream.executable.export public @dispatch
}

// CHECK-LABEL: util.global private @captured_value_if_ordering : !stream.resource<constant>
util.global private @captured_value_if_ordering : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Load parameter (will be in slice).
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"captured_param"> : vector<1024xi8>

  // Value defined outside scf.if but captured inside - this tests that
  // the slice ordering handles captured values correctly.
  %outside_value = arith.constant 42 : i32

  // The scf.if captures %outside_value and %param from outside.
  // When building the backward slice, we collect both the stored value
  // and captured values. The ordering must ensure %outside_value's producer
  // (arith.constant) comes before any op inside the region that uses it.
  %cond = arith.constant true
  // CHECK-NOT: scf.if
  %result = scf.if %cond -> !stream.resource<constant> {
    // Uses %outside_value (captured) and %param.
    %filled = stream.async.fill %outside_value, %param[%c0 to %c1024 for %c1024]
      : i32 -> %param as !stream.resource<constant>{%c1024}
    scf.yield %filled : !stream.resource<constant>
  } else {
    scf.yield %param : !stream.resource<constant>
  }

  // Encoder should transform this to load from encoded parameter.
  // CHECK: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %{{.+}}, @captured_value_if_ordering
  util.global.store %result, @captured_value_if_ordering : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests captured values with scf.for loop.
// Similar to the scf.if test but with loop-carried values.

stream.executable private @for_dispatch {
  stream.executable.export public @dispatch
}

// CHECK-LABEL: util.global private @captured_value_for_ordering : !stream.resource<constant>
util.global private @captured_value_for_ordering : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c256 = arith.constant 256 : index

  // Load parameter.
  %param = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"for_param"> : vector<1024xi8>

  // Value captured by the loop body.
  %fill_pattern = arith.constant 7 : i32

  // Loop that captures %fill_pattern from outside.
  // CHECK-NOT: scf.for
  %result = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %param) -> !stream.resource<constant> {
    // Uses captured %fill_pattern.
    %offset = arith.muli %i, %c256 : index
    %end = arith.addi %offset, %c256 : index
    %filled = stream.async.fill %fill_pattern, %acc[%offset to %end for %c256]
      : i32 -> %acc as !stream.resource<constant>{%c1024}
    scf.yield %filled : !stream.resource<constant>
  }

  // CHECK: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %{{.+}}, @captured_value_for_ordering
  util.global.store %result, @captured_value_for_ordering : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests nested scf.if with dispatch that uses multiple captured values.
// This more complex case exercises ordering across multiple region levels.

stream.executable private @nested_dispatch {
  stream.executable.export public @compute
}

// CHECK-LABEL: util.global private @nested_captured_ordering : !stream.resource<constant>
util.global private @nested_captured_ordering : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c2048 = arith.constant 2048 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Load two parameters that will both be used inside nested regions.
  %param_a = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"nested_a"> : vector<1024xi8>
  %param_b = stream.async.constant : !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"nested_b"> : vector<1024xi8>

  // Dispatch using both parameters - creates slice with multiple inputs.
  // CHECK-NOT: stream.async.dispatch
  %combined = stream.async.dispatch @nested_dispatch::@compute[%c1, %c1, %c1](
    %param_a[%c0 to %c1024 for %c1024],
    %param_b[%c0 to %c1024 for %c1024]
  ) : (!stream.resource<constant>{%c1024}, !stream.resource<constant>{%c1024}) -> !stream.resource<constant>{%c2048}

  // CHECK: stream.async.constant {{.+}} #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %{{.+}}, @nested_captured_ordering
  util.global.store %combined, @nested_captured_ordering : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests that an empty module with no parameter expressions runs cleanly.
// This verifies that when no output file is specified (the default) and
// no encoding work is found, the pass completes without errors.

// CHECK-LABEL: module {
// CHECK-NOT: module @encoder
module {
  // A simple global that doesn't involve any parameters.
  // CHECK: util.global private @no_params : i32
  util.global private @no_params : i32
  util.initializer {
    // CHECK: %[[C42:.+]] = arith.constant 42 : i32
    %c42 = arith.constant 42 : i32
    // CHECK: util.global.store %[[C42]], @no_params
    util.global.store %c42, @no_params : i32
    // CHECK: util.return
    util.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// External Timepoint Synchronization Tests
//===----------------------------------------------------------------------===//

// Tests that when a parameter load awaits on an external timepoint, the
// replacement async.constant also awaits on that timepoint.
// This exercises Source A of collectExternalTimepoints: external await
// timepoints from TimelineOpInterface ops in the expression.

// CHECK: module @encoder
// CHECK-LABEL: module {
// CHECK: util.global private @param_with_external_await : !stream.resource<constant>
util.global private @param_with_external_await : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c42_i32 = arith.constant 42 : i32

  // External timeline op that produces a timepoint we must wait on.
  // This is NOT part of the encoding expression (doesn't feed into the store).
  // CHECK: %[[EXTERNAL_RESOURCE:.+]], %[[EXTERNAL_TP:.+]] = stream.test.timeline_op
  %external_resource, %external_tp = stream.test.timeline_op
    with() : () -> !stream.resource<external>{%c1024} => !stream.timepoint

  // Parameter load that awaits on the external timepoint.
  // The expression starts here - this op and the fill below form the expression.
  %param = stream.async.constant await(%external_tp) :
    !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"awaiting_param"> : vector<1024xi8>

  // Transform the parameter so it gets hoisted.
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // The replacement should await on the external timepoint.
  // CHECK: %[[PARAM:.+]] = stream.async.constant await(%[[EXTERNAL_TP]])
  // CHECK-SAME: #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_with_external_await
  util.global.store %filled, @param_with_external_await : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}

// -----

// Tests that when a parameter load awaits on a joined timepoint from multiple
// external timeline ops, the replacement async.constant awaits on that same
// joined timepoint. This exercises the case where the join is in the expression
// slice but is not a resource contributor (it only produces a timepoint).

// CHECK: module @encoder
// CHECK-LABEL: module {
// CHECK: util.global private @param_with_joined_external_timepoints : !stream.resource<constant>
util.global private @param_with_joined_external_timepoints : !stream.resource<constant>

// CHECK: util.initializer {
util.initializer {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c42_i32 = arith.constant 42 : i32

  // Two external timeline ops that produce timepoints we must wait on.
  // Their resources are unused, so they're not resource contributors.
  // CHECK-DAG: %[[EXT_R1:.+]], %[[EXT_TP1:.+]] = stream.test.timeline_op
  %ext_r1, %ext_tp1 = stream.test.timeline_op
    with() : () -> !stream.resource<external>{%c1024} => !stream.timepoint
  // CHECK-DAG: %[[EXT_R2:.+]], %[[EXT_TP2:.+]] = stream.test.timeline_op
  %ext_r2, %ext_tp2 = stream.test.timeline_op
    with() : () -> !stream.resource<external>{%c1024} => !stream.timepoint

  // Join the timepoints. The join is in the expression but doesn't contribute
  // resources, so its result timepoint should be considered external.
  // CHECK: %[[JOINED_TP:.+]] = stream.timepoint.join max(%[[EXT_TP1]], %[[EXT_TP2]]) => !stream.timepoint
  %joined_tp = stream.timepoint.join max(%ext_tp1, %ext_tp2) => !stream.timepoint

  // Parameter load that awaits on the joined timepoint.
  %param = stream.async.constant await(%joined_tp) :
    !stream.resource<constant>{%c1024} =
    #stream.parameter.named<"model"::"joined_await_param"> : vector<1024xi8>

  // Transform the parameter so it gets hoisted.
  // CHECK-NOT: stream.async.fill
  %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] :
    i32 -> %param as !stream.resource<constant>{%c1024}

  // The replacement should await on the same joined timepoint.
  // CHECK: %[[PARAM:.+]] = stream.async.constant await(%[[JOINED_TP]])
  // CHECK-SAME: #stream.parameter.named<""::"parameter0">
  // CHECK: util.global.store %[[PARAM]], @param_with_joined_external_timepoints
  util.global.store %filled, @param_with_joined_external_timepoints : !stream.resource<constant>
  // CHECK: util.return
  util.return
  // CHECK: }
}
