// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=consolidate output-scope=encoded' %s | FileCheck %s --check-prefixes=CONSOLIDATE,CONSOLIDATE-HOIST --implicit-check-not=stream.async.parameter.load --implicit-check-not=stream.async.parameter.scatter --implicit-check-not=stream.async.fill --implicit-check-not='#stream.parameter.named<"model"::'
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=consolidate output-scope=encoded hoist-parameter-expressions=false' %s | FileCheck %s --check-prefixes=CONSOLIDATE,CONSOLIDATE-NOHOIST --implicit-check-not=stream.async.parameter.load --implicit-check-not=stream.async.parameter.scatter --implicit-check-not=stream.async.fill --implicit-check-not='#stream.parameter.named<"model"::'

// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay output-scope=encoded' %s | FileCheck %s --check-prefixes=OVERLAY,OVERLAY-HOIST --implicit-check-not='module @encoder' --implicit-check-not=stream.async.parameter.load --implicit-check-not=stream.async.parameter.scatter --implicit-check-not=stream.async.fill --implicit-check-not='#stream.parameter.named<"encoded"::'
// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=overlay output-scope=encoded hoist-parameter-expressions=false' %s | FileCheck %s --check-prefixes=OVERLAY,OVERLAY-NOHOIST --implicit-check-not='module @encoder' --implicit-check-not=stream.async.parameter.load --implicit-check-not=stream.async.parameter.scatter --implicit-check-not=stream.async.fill --implicit-check-not='#stream.parameter.named<"encoded"::'

// A parameter load shared by a global store and a runtime consumer occupies
// one archive range in consolidate mode. Both consumers use the same resource.
// Overlay mode keeps the original load and does not produce an encoder.
// CONSOLIDATE-LABEL: module @encoder
// CONSOLIDATE: util.func public @__encode_parameter_indices_all()
// CONSOLIDATE: %[[INDEX_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER:parameter[0-9]+]]"
// CONSOLIDATE: %[[SIZE:.+]] = arith.constant 1024 : i64
// CONSOLIDATE: %[[ENTRY:.+]] = util.list.construct [{{.*}}%[[INDEX_KEY]] : !util.buffer, {{.*}}, %[[SIZE]] : i64, {{.*}}]
// CONSOLIDATE: util.list.construct [%[[ENTRY]] : !util.list<?>]
// CONSOLIDATE-LABEL: util.func public @__encode_parameters_all(
// CONSOLIDATE: %[[MODEL:.+]] = util.buffer.constant : !util.buffer = "model"
// CONSOLIDATE: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weight"
// CONSOLIDATE: %[[ZERO:.+]] = arith.constant 0 : i64
// CONSOLIDATE: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[ZERO]]]
// CONSOLIDATE: %[[OUTPUT_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER]]"
// CONSOLIDATE: %[[OUTPUT_SCOPE:.+]] = util.buffer.constant : !util.buffer = "encoded"
// CONSOLIDATE: stream.async.parameter.scatter
// CONSOLIDATE-NEXT: {{.*}} -> %[[OUTPUT_SCOPE]]::%[[OUTPUT_KEY]][%[[ZERO]]]
// CONSOLIDATE: util.return

// CONSOLIDATE-LABEL: module @shared_load
// CONSOLIDATE: util.initializer
// CONSOLIDATE: %[[PARAMETER:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"encoded"::"[[$PARAMETER]]"> : tensor<1024xi8>
// CONSOLIDATE-NEXT: util.global.store %[[PARAMETER]], @weight
// CONSOLIDATE-NEXT: %[[RUNTIME:.+]] = util.optimization_barrier %[[PARAMETER]]
// CONSOLIDATE-NEXT: util.global.store %[[RUNTIME]], @runtime_weight
// CONSOLIDATE-NEXT: util.return

// OVERLAY-LABEL: module @shared_load
// OVERLAY: util.initializer
// OVERLAY: %[[SOURCE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"model"::"weight"> : vector<1024xi8>
// OVERLAY-NEXT: util.global.store %[[SOURCE]], @weight
// OVERLAY-NEXT: %[[RUNTIME:.+]] = util.optimization_barrier %[[SOURCE]]
// OVERLAY-NEXT: util.global.store %[[RUNTIME]], @runtime_weight
// OVERLAY-NEXT: util.return
module @shared_load {
  util.global private @weight : !stream.resource<constant>
  util.global private @runtime_weight : !stream.resource<constant>
  util.initializer {
    %c1024 = arith.constant 1024 : index
    %param = stream.async.constant : !stream.resource<constant>{%c1024} =
      #stream.parameter.named<"model"::"weight"> : vector<1024xi8>
    util.global.store %param, @weight : !stream.resource<constant>
    %runtime = util.optimization_barrier %param : !stream.resource<constant>
    util.global.store %runtime, @runtime_weight : !stream.resource<constant>
    util.return
  }
}

// -----

// A parameter load can feed a simple store to a global, an offline transformation,
// and a transformation that gets rejected by the hoisting policy because of the expanded
// size.
// With hoisting and consolidate mode enabled, the archive holds the original bytes once
// alongside the offline result. No duplication is expected for the global store of the unmodified value.
// With hoisting disabled, the original bytes still get serialized because of the consolidate mode.

// In consolidate mode with hoisting enabled, encode the fill and the shared input.
// CONSOLIDATE-HOIST-LABEL: module @encoder
// CONSOLIDATE-HOIST: util.func public @__encode_parameter_indices_all()
// CONSOLIDATE-HOIST: %[[INDEX_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER:parameter[0-9]+]]"
// CONSOLIDATE-HOIST: %[[SIZE:.+]] = arith.constant 2048 : i64
// CONSOLIDATE-HOIST: %[[ENTRY:.+]] = util.list.construct [{{.*}}%[[INDEX_KEY]] : !util.buffer, {{.*}}, %[[SIZE]] : i64, {{.*}}]
// CONSOLIDATE-HOIST: util.list.construct [%[[ENTRY]] : !util.list<?>]
// CONSOLIDATE-HOIST-LABEL: util.func public @__encode_parameters_all(
// CONSOLIDATE-HOIST: %[[MODEL:.+]] = util.buffer.constant : !util.buffer = "model"
// CONSOLIDATE-HOIST: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weight"
// CONSOLIDATE-HOIST: %[[ZERO:.+]] = arith.constant 0 : i64
// CONSOLIDATE-HOIST: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[ZERO]]]
// CONSOLIDATE-HOIST: stream.async.fill
// CONSOLIDATE-HOIST: %[[OUTPUT_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER]]"
// CONSOLIDATE-HOIST: %[[OUTPUT_SCOPE:.+]] = util.buffer.constant : !util.buffer = "encoded"
// CONSOLIDATE-HOIST: stream.async.parameter.scatter
// CONSOLIDATE-HOIST-NEXT: {{.*}} -> %[[OUTPUT_SCOPE]]::%[[OUTPUT_KEY]][%[[ZERO]]]
// CONSOLIDATE-HOIST: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[ZERO]]]
// CONSOLIDATE-HOIST: %[[OFFSET:.+]] = arith.constant 1024 : i64
// CONSOLIDATE-HOIST: stream.async.parameter.scatter
// CONSOLIDATE-HOIST-NEXT: {{.*}} -> %[[OUTPUT_SCOPE]]::%[[OUTPUT_KEY]][%[[OFFSET]]]
// CONSOLIDATE-HOIST: util.return

// CONSOLIDATE-HOIST-LABEL: module @shared_load_with_encodings
// CONSOLIDATE-HOIST: util.initializer
// CONSOLIDATE-HOIST-DAG: %[[SIZE:.+]] = arith.constant 2048 : index
// CONSOLIDATE-HOIST-DAG: %[[ZERO:.+]] = arith.constant 0 : index
// CONSOLIDATE-HOIST-DAG: %[[LENGTH:.+]] = arith.constant 1024 : index
// CONSOLIDATE-HOIST: %[[STORAGE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"encoded"::"[[$PARAMETER]]"> : tensor<2048xi8>
// CONSOLIDATE-HOIST-NEXT: %[[SOURCE:.+]] = stream.resource.subview %[[STORAGE]][%[[LENGTH]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[LENGTH]]}
// CONSOLIDATE-HOIST-NEXT: %[[TRANSFERRED:.+]] = stream.async.transfer %[[SOURCE]]
// CONSOLIDATE-HOIST-NEXT: util.global.store %[[TRANSFERRED]], @weight
// CONSOLIDATE-HOIST: %[[FILLED_STORAGE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"encoded"::"[[$PARAMETER]]"> : tensor<2048xi8>
// CONSOLIDATE-HOIST-NEXT: %[[FILLED:.+]] = stream.resource.subview %[[FILLED_STORAGE]][%[[ZERO]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[LENGTH]]}
// CONSOLIDATE-HOIST-NEXT: util.global.store %[[FILLED]], @filled_weight
// CONSOLIDATE-HOIST-NEXT: %[[PADDED:.+]] = stream.async.splat
// CONSOLIDATE-HOIST-NEXT: %[[EXPANDED:.+]] = stream.async.update %[[SOURCE]], %[[PADDED]][
// CONSOLIDATE-HOIST-NEXT: util.global.store %[[EXPANDED]], @expanded_weight
// CONSOLIDATE-HOIST-NEXT: util.return

// In consolidate mode with hoisting disabled, encode only the shared input.
// CONSOLIDATE-NOHOIST-LABEL: module @encoder
// CONSOLIDATE-NOHOIST: util.func public @__encode_parameter_indices_all()
// CONSOLIDATE-NOHOIST: %[[INDEX_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER:parameter[0-9]+]]"
// CONSOLIDATE-NOHOIST: %[[SIZE:.+]] = arith.constant 1024 : i64
// CONSOLIDATE-NOHOIST: %[[ENTRY:.+]] = util.list.construct [{{.*}}%[[INDEX_KEY]] : !util.buffer, {{.*}}, %[[SIZE]] : i64, {{.*}}]
// CONSOLIDATE-NOHOIST: util.list.construct [%[[ENTRY]] : !util.list<?>]
// CONSOLIDATE-NOHOIST-LABEL: util.func public @__encode_parameters_all(
// CONSOLIDATE-NOHOIST: %[[MODEL:.+]] = util.buffer.constant : !util.buffer = "model"
// CONSOLIDATE-NOHOIST: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weight"
// CONSOLIDATE-NOHOIST: %[[ZERO:.+]] = arith.constant 0 : i64
// CONSOLIDATE-NOHOIST: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[ZERO]]]
// CONSOLIDATE-NOHOIST: %[[OUTPUT_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER]]"
// CONSOLIDATE-NOHOIST: %[[OUTPUT_SCOPE:.+]] = util.buffer.constant : !util.buffer = "encoded"
// CONSOLIDATE-NOHOIST: stream.async.parameter.scatter
// CONSOLIDATE-NOHOIST-NEXT: {{.*}} -> %[[OUTPUT_SCOPE]]::%[[OUTPUT_KEY]][%[[ZERO]]]
// CONSOLIDATE-NOHOIST: util.return

// CONSOLIDATE-NOHOIST-LABEL: module @shared_load_with_encodings
// CONSOLIDATE-NOHOIST: util.initializer
// CONSOLIDATE-NOHOIST: %[[SOURCE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"encoded"::"[[$PARAMETER]]"> : tensor<1024xi8>
// CONSOLIDATE-NOHOIST-NEXT: %[[TRANSFERRED:.+]] = stream.async.transfer %[[SOURCE]]
// CONSOLIDATE-NOHOIST-NEXT: util.global.store %[[TRANSFERRED]], @weight
// CONSOLIDATE-NOHOIST: %[[FILLED:.+]] = stream.async.fill {{.*}}, %[[SOURCE]][
// CONSOLIDATE-NOHOIST-NEXT: util.global.store %[[FILLED]], @filled_weight
// CONSOLIDATE-NOHOIST-NEXT: %[[PADDED:.+]] = stream.async.splat
// CONSOLIDATE-NOHOIST-NEXT: %[[EXPANDED:.+]] = stream.async.update %[[SOURCE]], %[[PADDED]][
// CONSOLIDATE-NOHOIST-NEXT: util.global.store %[[EXPANDED]], @expanded_weight
// CONSOLIDATE-NOHOIST-NEXT: util.return

// In overlay mode with hoisting enabled, serialize only the filled result.
// The unchanged global and the runtime expansion keep sharing the original load.
// OVERLAY-HOIST-LABEL: module @encoder
// OVERLAY-HOIST: util.func public @__encode_parameter_indices_all()
// OVERLAY-HOIST: %[[INDEX_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER:parameter[0-9]+]]"
// OVERLAY-HOIST: %[[SIZE:.+]] = arith.constant 1024 : i64
// OVERLAY-HOIST: %[[ENTRY:.+]] = util.list.construct [{{.*}}%[[INDEX_KEY]] : !util.buffer, {{.*}}, %[[SIZE]] : i64, {{.*}}]
// OVERLAY-HOIST: util.list.construct [%[[ENTRY]] : !util.list<?>]
// OVERLAY-HOIST-LABEL: util.func public @__encode_parameters_all(
// OVERLAY-HOIST: %[[MODEL:.+]] = util.buffer.constant : !util.buffer = "model"
// OVERLAY-HOIST: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weight"
// OVERLAY-HOIST: %[[ZERO:.+]] = arith.constant 0 : i64
// OVERLAY-HOIST: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[ZERO]]]
// OVERLAY-HOIST: stream.async.fill
// OVERLAY-HOIST: %[[OUTPUT_KEY:.+]] = util.buffer.constant : !util.buffer = "[[$PARAMETER]]"
// OVERLAY-HOIST: %[[OUTPUT_SCOPE:.+]] = util.buffer.constant : !util.buffer = "encoded"
// OVERLAY-HOIST: stream.async.parameter.scatter
// OVERLAY-HOIST-NEXT: {{.*}} -> %[[OUTPUT_SCOPE]]::%[[OUTPUT_KEY]][%[[ZERO]]]
// OVERLAY-HOIST: util.return

// OVERLAY-HOIST-LABEL: module @shared_load_with_encodings
// OVERLAY-HOIST: util.initializer
// OVERLAY-HOIST: %[[SOURCE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"model"::"weight"> : vector<1024xi8>
// OVERLAY-HOIST-NEXT: %[[TRANSFERRED:.+]] = stream.async.transfer %[[SOURCE]]
// OVERLAY-HOIST-NEXT: util.global.store %[[TRANSFERRED]], @weight
// OVERLAY-HOIST-NEXT: %[[FILLED:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"encoded"::"[[$PARAMETER]]"> : tensor<1024xi8>
// OVERLAY-HOIST-NEXT: util.global.store %[[FILLED]], @filled_weight
// OVERLAY-HOIST-NEXT: %[[PADDED:.+]] = stream.async.splat
// OVERLAY-HOIST-NEXT: %[[EXPANDED:.+]] = stream.async.update %[[SOURCE]], %[[PADDED]][
// OVERLAY-HOIST-NEXT: util.global.store %[[EXPANDED]], @expanded_weight
// OVERLAY-HOIST-NEXT: util.return

// In overlay mode with hoisting disabled, produce no encoder. Both
// transformations stay at runtime and share the original load with the global.
// OVERLAY-NOHOIST-LABEL: module @shared_load_with_encodings
// OVERLAY-NOHOIST: util.initializer
// OVERLAY-NOHOIST: %[[SOURCE:.+]] = stream.async.constant {{.*}} #stream.parameter.named<"model"::"weight"> : vector<1024xi8>
// OVERLAY-NOHOIST-NEXT: %[[TRANSFERRED:.+]] = stream.async.transfer %[[SOURCE]]
// OVERLAY-NOHOIST-NEXT: util.global.store %[[TRANSFERRED]], @weight
// OVERLAY-NOHOIST-NEXT: %[[FILLED:.+]] = stream.async.fill {{.*}}, %[[SOURCE]][
// OVERLAY-NOHOIST-NEXT: util.global.store %[[FILLED]], @filled_weight
// OVERLAY-NOHOIST-NEXT: %[[PADDED:.+]] = stream.async.splat
// OVERLAY-NOHOIST-NEXT: %[[EXPANDED:.+]] = stream.async.update %[[SOURCE]], %[[PADDED]][
// OVERLAY-NOHOIST-NEXT: util.global.store %[[EXPANDED]], @expanded_weight
// OVERLAY-NOHOIST-NEXT: util.return
module @shared_load_with_encodings {
  util.global private @weight : !stream.resource<constant>
  util.global private @filled_weight : !stream.resource<constant>
  util.global private @expanded_weight : !stream.resource<constant>
  util.initializer {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32
    %param = stream.async.constant : !stream.resource<constant>{%c1024} =
      #stream.parameter.named<"model"::"weight"> : vector<1024xi8>
    %transferred = stream.async.transfer %param :
      !stream.resource<constant>{%c1024} -> !stream.resource<constant>{%c1024}
    util.global.store %transferred, @weight : !stream.resource<constant>
    %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] :
      i32 -> %param as !stream.resource<constant>{%c1024}
    util.global.store %filled, @filled_weight : !stream.resource<constant>
    %padded = stream.async.splat %c0_i32 : i32 -> !stream.resource<constant>{%c2048}
    %expanded = stream.async.update %param, %padded[%c0 to %c1024] :
      !stream.resource<constant>{%c1024} -> %padded as !stream.resource<constant>{%c2048}
    util.global.store %expanded, @expanded_weight : !stream.resource<constant>
    util.return
  }
}
