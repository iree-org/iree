// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ElideUnusedParameterLoad
util.func private @ElideUnusedParameterLoad() {
  %c0 = arith.constant 0 : i64
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.async.parameter.load
  %unused, %unused_ready = stream.async.parameter.load "scope"::"key"[%c0] : !stream.resource<constant>{%c100} => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @FoldAsyncParameterReadTargetSubview
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[TARGET_SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldAsyncParameterReadTargetSubview(%target: !stream.resource<transient>, %target_size: index, %offset: index, %length: index) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %param_offset = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %resource_offset = arith.constant 100 : index
  %resource_end = arith.constant 300 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %transfer_length = arith.constant 200 : index
  // CHECK-DAG: %[[RESOURCE_END:.+]] = arith.constant 300 : index
  %subview_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%target_size} -> !stream.resource<transient>{%subview_size}
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read "scope"::"key"[%[[FOLDED_PARAM_OFFSET]]] -> %[[TARGET]][%[[FOLDED_RESOURCE_OFFSET]] to %[[FOLDED_RESOURCE_END]] for %[[TRANSFER_LENGTH]]] : !stream.resource<transient>{%[[TARGET_SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.read "scope"::"key"[%param_offset] -> %subview[%resource_offset to %resource_end for %transfer_length] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[TARGET_SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%target_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterWriteSourceSubview
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SOURCE_SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @FoldAsyncParameterWriteSourceSubview(%source: !stream.resource<transient>, %source_size: index, %offset: index, %length: index) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %param_offset = arith.constant 50 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET:.+]] = arith.constant 100 : index
  %resource_offset = arith.constant 100 : index
  %resource_end = arith.constant 300 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH:.+]] = arith.constant 200 : index
  %transfer_length = arith.constant 200 : index
  // CHECK-DAG: %[[RESOURCE_END:.+]] = arith.constant 300 : index
  %subview_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%source_size} -> !stream.resource<transient>{%subview_size}
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.write %[[SOURCE]][%[[FOLDED_RESOURCE_OFFSET]] to %[[FOLDED_RESOURCE_END]] for %[[TRANSFER_LENGTH]]] -> "scope"::"key"[%[[FOLDED_PARAM_OFFSET]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} => !stream.timepoint
  %result, %result_ready = stream.async.parameter.write %subview[%resource_offset to %resource_end for %transfer_length] -> "scope"::"key"[%param_offset] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SOURCE_SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%source_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterGatherTargetSubview
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[TARGET_SIZE:.+]]: index, %[[OFFSET:.+]]: index)
util.func private @FoldAsyncParameterGatherTargetSubview(%target: !stream.resource<transient>, %target_size: index, %offset: index) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  %param_offset0 = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  %param_offset1 = arith.constant 51 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET0:.+]] = arith.constant 100 : index
  %resource_offset0 = arith.constant 100 : index
  // CHECK-DAG: %[[RESOURCE_OFFSET1:.+]] = arith.constant 101 : index
  %resource_offset1 = arith.constant 101 : index
  // CHECK-DAG: %[[RESOURCE_END0:.+]] = arith.constant 300 : index
  %resource_end0 = arith.constant 300 : index
  // CHECK-DAG: %[[RESOURCE_END1:.+]] = arith.constant 302 : index
  %resource_end1 = arith.constant 302 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH0:.+]] = arith.constant 200 : index
  %transfer_length0 = arith.constant 200 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH1:.+]] = arith.constant 201 : index
  %transfer_length1 = arith.constant 201 : index
  %subview_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET0:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET1:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET0:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET1:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END0:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END0]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END1:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END1]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%offset] : !stream.resource<transient>{%target_size} -> !stream.resource<transient>{%subview_size}
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: "scope"::"key0"[%[[FOLDED_PARAM_OFFSET0]]] -> %[[TARGET]][%[[FOLDED_RESOURCE_OFFSET0]] to %[[FOLDED_RESOURCE_END0]] for %[[TRANSFER_LENGTH0]]] : !stream.resource<transient>{%[[TARGET_SIZE]]},
  // CHECK-NEXT: "scope"::"key1"[%[[FOLDED_PARAM_OFFSET1]]] -> %[[TARGET]][%[[FOLDED_RESOURCE_OFFSET1]] to %[[FOLDED_RESOURCE_END1]] for %[[TRANSFER_LENGTH1]]] : !stream.resource<transient>{%[[TARGET_SIZE]]}
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.gather {
    "scope"::"key0"[%param_offset0] -> %subview[%resource_offset0 to %resource_end0 for %transfer_length0] : !stream.resource<transient>{%subview_size},
    "scope"::"key1"[%param_offset1] -> %subview[%resource_offset1 to %resource_end1 for %transfer_length1] : !stream.resource<transient>{%subview_size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[TARGET_SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%target_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterScatterSourceSubview
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<transient>, %[[SOURCE_SIZE:.+]]: index, %[[OFFSET:.+]]: index)
util.func private @FoldAsyncParameterScatterSourceSubview(%source: !stream.resource<transient>, %source_size: index, %offset: index) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET0:.+]] = arith.constant 50 : i64
  %param_offset0 = arith.constant 50 : i64
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 51 : i64
  %param_offset1 = arith.constant 51 : i64
  // CHECK-DAG: %[[RESOURCE_OFFSET0:.+]] = arith.constant 100 : index
  %resource_offset0 = arith.constant 100 : index
  // CHECK-DAG: %[[RESOURCE_OFFSET1:.+]] = arith.constant 101 : index
  %resource_offset1 = arith.constant 101 : index
  // CHECK-DAG: %[[RESOURCE_END0:.+]] = arith.constant 300 : index
  %resource_end0 = arith.constant 300 : index
  // CHECK-DAG: %[[RESOURCE_END1:.+]] = arith.constant 302 : index
  %resource_end1 = arith.constant 302 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH0:.+]] = arith.constant 200 : index
  %transfer_length0 = arith.constant 200 : index
  // CHECK-DAG: %[[TRANSFER_LENGTH1:.+]] = arith.constant 201 : index
  %transfer_length1 = arith.constant 201 : index
  %subview_size = arith.constant 300 : index
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET0:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_PARAM_OFFSET1:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET0:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET0]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_OFFSET1:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END0:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END0]]
  // CHECK-DAG: %[[FOLDED_RESOURCE_END1:.+]] = arith.addi %[[OFFSET]], %[[RESOURCE_END1]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %source[%offset] : !stream.resource<transient>{%source_size} -> !stream.resource<transient>{%subview_size}
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.scatter {
  // CHECK-NEXT: %[[SOURCE]][%[[FOLDED_RESOURCE_OFFSET0]] to %[[FOLDED_RESOURCE_END0]] for %[[TRANSFER_LENGTH0]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> "scope"::"key0"[%[[FOLDED_PARAM_OFFSET0]]],
  // CHECK-NEXT: %[[SOURCE]][%[[FOLDED_RESOURCE_OFFSET1]] to %[[FOLDED_RESOURCE_END1]] for %[[TRANSFER_LENGTH1]]] : !stream.resource<transient>{%[[SOURCE_SIZE]]} -> "scope"::"key1"[%[[FOLDED_PARAM_OFFSET1]]]
  // CHECK-NEXT: } : !stream.resource<transient> => !stream.timepoint
  %result, %result_ready = stream.async.parameter.scatter {
    %subview[%resource_offset0 to %resource_end0 for %transfer_length0] : !stream.resource<transient>{%subview_size} -> "scope"::"key0"[%param_offset0],
    %subview[%resource_offset1 to %resource_end1 for %transfer_length1] : !stream.resource<transient>{%subview_size} -> "scope"::"key1"[%param_offset1]
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SOURCE_SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%source_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSubview
// CHECK-SAME: (%[[OFFSET:.+]]: index)
util.func private @FoldAsyncParameterLoadResultSubview(%offset: index) -> !stream.resource<constant> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 50 : i64
  %param_offset = arith.constant 50 : i64
  %load_size = arith.constant 500 : index
  // CHECK-DAG: %[[SUBVIEW_OFFSET:.+]] = arith.constant 100 : index
  %subview_offset = arith.constant 100 : index
  // CHECK-DAG: %[[SUBVIEW_SIZE:.+]] = arith.constant 200 : index
  %subview_size = arith.constant 200 : index
  // CHECK-DAG: %[[DYNAMIC_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[SUBVIEW_OFFSET]]
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[DYNAMIC_OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[FOLDED_OFFSET]]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  %dynamic_offset = arith.addi %offset, %subview_offset : index
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %loaded[%dynamic_offset] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%subview_size}
  // CHECK: %[[LOADED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  %subview_sync = stream.timepoint.await %loaded_ready => %subview : !stream.resource<constant>{%subview_size}
  // CHECK: util.return %[[LOADED_SYNC]]
  util.return %subview_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSlice
// CHECK-SAME: (%[[OFFSET:.+]]: index)
util.func private @FoldAsyncParameterLoadResultSlice(%offset: index) -> !stream.resource<constant> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 60 : i64
  %param_offset = arith.constant 60 : i64
  %load_size = arith.constant 600 : index
  // CHECK-DAG: %[[SLICE_OFFSET:.+]] = arith.constant 110 : index
  %slice_offset = arith.constant 110 : index
  %slice_end = arith.constant 310 : index
  // CHECK-DAG: %[[SLICE_SIZE:.+]] = arith.constant 210 : index
  %slice_size = arith.constant 210 : index
  // CHECK-DAG: %[[DYNAMIC_OFFSET:.+]] = arith.addi %[[OFFSET]], %[[SLICE_OFFSET]]
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[DYNAMIC_OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[FOLDED_OFFSET]]] : !stream.resource<constant>{%[[SLICE_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%load_size}
  %dynamic_offset = arith.addi %offset, %slice_offset : index
  %dynamic_end = arith.addi %dynamic_offset, %slice_size : index
  // CHECK-NOT: stream.async.slice
  %sliced = stream.async.slice %awaited[%dynamic_offset to %dynamic_end] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%slice_size}
  // CHECK: %[[SLICED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SLICE_SIZE]]}
  %sliced_sync = stream.timepoint.await %loaded_ready => %sliced : !stream.resource<constant>{%slice_size}
  // CHECK: util.return %[[SLICED_SYNC]]
  util.return %sliced_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSubviewMultipleAwaits
util.func private @FoldAsyncParameterLoadResultSubviewMultipleAwaits() -> (!stream.resource<constant>, !stream.resource<constant>) {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 0 : i64
  %c0_i64 = arith.constant 0 : i64
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[SUBVIEW_SIZE:.+]] = arith.constant 50 : index
  %c50 = arith.constant 50 : index
  // CHECK-DAG: %[[SUBVIEW_OFFSET:.+]] = arith.constant 10 : index
  %c10 = arith.constant 10 : index
  // Load result awaited once, but await result has multiple different uses.
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[PARAM_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]}
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SIZE]]}
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%c100}
  // Subview should NOT fold because awaited result has multiple uses (direct return + subview).
  // CHECK: %[[SUBVIEW:.+]] = stream.resource.subview %[[AWAITED]][%[[SUBVIEW_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  %subview = stream.resource.subview %awaited[%c10] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c50}
  // CHECK: util.return %[[AWAITED]], %[[SUBVIEW]]
  util.return %awaited, %subview : !stream.resource<constant>, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSubviewMultipleSubviews
util.func private @FoldAsyncParameterLoadResultSubviewMultipleSubviews() -> (!stream.resource<constant>, !stream.resource<constant>) {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 0 : i64
  %c0_i64 = arith.constant 0 : i64
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-DAG: %[[SUBVIEW1_SIZE:.+]] = arith.constant 50 : index
  %c50 = arith.constant 50 : index
  // CHECK-DAG: %[[SUBVIEW2_SIZE:.+]] = arith.constant 40 : index
  %c40 = arith.constant 40 : index
  // CHECK-DAG: %[[SUBVIEW1_OFFSET:.+]] = arith.constant 10 : index
  %c10 = arith.constant 10 : index
  // CHECK-DAG: %[[SUBVIEW2_OFFSET:.+]] = arith.constant 20 : index
  %c20 = arith.constant 20 : index
  // Load result awaited once, but await result used by multiple different subviews.
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[PARAM_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]}
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%c0_i64] : !stream.resource<constant>{%c100} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SIZE]]}
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%c100}
  // Both subviews should NOT fold because awaited result has multiple uses.
  // CHECK: %[[SUBVIEW1:.+]] = stream.resource.subview %[[AWAITED]][%[[SUBVIEW1_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW1_SIZE]]}
  %subview1 = stream.resource.subview %awaited[%c10] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c50}
  // CHECK: %[[SUBVIEW2:.+]] = stream.resource.subview %[[AWAITED]][%[[SUBVIEW2_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[SUBVIEW2_SIZE]]}
  %subview2 = stream.resource.subview %awaited[%c20] : !stream.resource<constant>{%c100} -> !stream.resource<constant>{%c40}
  // CHECK: util.return %[[SUBVIEW1]], %[[SUBVIEW2]]
  util.return %subview1, %subview2 : !stream.resource<constant>, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSliceMultipleSlices
// CHECK-SAME: (%[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[SIZE:[a-zA-Z0-9]+]]: index, %[[SLICE1_SIZE:[a-zA-Z0-9]+]]: index, %[[SLICE2_SIZE:[a-zA-Z0-9]+]]: index, %[[SLICE1_START:[a-zA-Z0-9]+]]: index, %[[SLICE1_END:[a-zA-Z0-9]+]]: index, %[[SLICE2_START:[a-zA-Z0-9]+]]: index, %[[SLICE2_END:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterLoadResultSliceMultipleSlices(%param_offset: i64, %size: index, %slice1_size: index, %slice2_size: index, %slice1_start: index, %slice1_end: index, %slice2_start: index, %slice2_end: index) -> (!stream.resource<constant>, !stream.resource<constant>) {
  // Load result awaited, but await result used by multiple different slices.
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[PARAM_OFFSET]]] : !stream.resource<constant>{%[[SIZE]]}
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%size} => !stream.timepoint
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SIZE]]}
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%size}
  // Both slices should NOT fold because awaited result has multiple uses.
  // CHECK: %[[SLICED1:.+]] = stream.async.slice %[[AWAITED]][%[[SLICE1_START]] to %[[SLICE1_END]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[SLICE1_SIZE]]}
  %sliced1 = stream.async.slice %awaited[%slice1_start to %slice1_end] : !stream.resource<constant>{%size} -> !stream.resource<constant>{%slice1_size}
  // CHECK: %[[SLICED2:.+]] = stream.async.slice %[[AWAITED]][%[[SLICE2_START]] to %[[SLICE2_END]]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<constant>{%[[SLICE2_SIZE]]}
  %sliced2 = stream.async.slice %awaited[%slice2_start to %slice2_end] : !stream.resource<constant>{%size} -> !stream.resource<constant>{%slice2_size}
  // CHECK: util.return %[[SLICED1]], %[[SLICED2]]
  util.return %sliced1, %sliced2 : !stream.resource<constant>, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultZeroOffsetSubview
// CHECK-SAME: (%[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[LOAD_SIZE:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterLoadResultZeroOffsetSubview(%param_offset: i64, %load_size: index, %subview_offset: index, %subview_size: index) -> !stream.resource<constant> {
  // Zero offset subview is identity, but should still fold.
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_OFFSET:.+]] = arith.addi %[[PARAM_OFFSET]], %[[OFFSET_I64]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[ADJUSTED_OFFSET]]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %loaded[%subview_offset] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%subview_size}
  // CHECK: %[[LOADED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  %subview_sync = stream.timepoint.await %loaded_ready => %subview : !stream.resource<constant>{%subview_size}
  // CHECK: util.return %[[LOADED_SYNC]]
  util.return %subview_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultLargeI64Offsets
// CHECK-SAME: (%[[LARGE_PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[LOAD_SIZE:[a-zA-Z0-9]+]]: index, %[[LARGE_SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterLoadResultLargeI64Offsets(%large_param_offset: i64, %load_size: index, %large_subview_offset: index, %subview_size: index) -> !stream.resource<constant> {
  // Test with very large i64 offsets to verify arithmetic correctness.
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[LARGE_SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_OFFSET:.+]] = arith.addi %[[LARGE_PARAM_OFFSET]], %[[OFFSET_I64]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[ADJUSTED_OFFSET]]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%large_param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %loaded[%large_subview_offset] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%subview_size}
  // CHECK: %[[LOADED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  %subview_sync = stream.timepoint.await %loaded_ready => %subview : !stream.resource<constant>{%subview_size}
  // CHECK: util.return %[[LOADED_SYNC]]
  util.return %subview_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterLoadResultSliceDynamicBoth
// CHECK-SAME: (%[[DYNAMIC_START:.+]]: index, %[[DYNAMIC_END:.+]]: index)
util.func private @FoldAsyncParameterLoadResultSliceDynamicBoth(%dynamic_start: index, %dynamic_end: index) -> !stream.resource<constant> {
  // CHECK-DAG: %[[PARAM_OFFSET:.+]] = arith.constant 250 : i64
  %param_offset = arith.constant 250 : i64
  %load_size = arith.constant 1000 : index
  // CHECK-DAG: %[[SLICE_SIZE:.+]] = arith.subi %[[DYNAMIC_END]], %[[DYNAMIC_START]]
  %slice_size = arith.subi %dynamic_end, %dynamic_start : index
  // Both start and end are dynamic.
  // CHECK-DAG: %[[START_I64:.+]] = arith.index_cast %[[DYNAMIC_START]] : index to i64
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[START_I64]], %[[PARAM_OFFSET]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[FOLDED_OFFSET]]] : !stream.resource<constant>{%[[SLICE_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.timepoint.await
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%load_size}
  // CHECK-NOT: stream.async.slice
  %sliced = stream.async.slice %awaited[%dynamic_start to %dynamic_end] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%slice_size}
  // CHECK: %[[SLICED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SLICE_SIZE]]}
  %sliced_sync = stream.timepoint.await %loaded_ready => %sliced : !stream.resource<constant>{%slice_size}
  // CHECK: util.return %[[SLICED_SYNC]]
  util.return %sliced_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterLoadWait
util.func private @ElideImmediateAsyncParameterLoadWait() -> (!stream.resource<constant>, !stream.timepoint) {
  %c0 = arith.constant 0 : i64
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: await
  // CHECK: stream.async.parameter.load "scope"::"key"
  %result, %result_ready = stream.async.parameter.load await(%imm) "scope"::"key"[%c0] : !stream.resource<constant>{%c100} => !stream.timepoint
  util.return %result, %result_ready : !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterReadWait
util.func private @ElideImmediateAsyncParameterReadWait(%target: !stream.resource<transient>, %target_size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : i64
  %c0_index = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: await
  // CHECK: stream.async.parameter.read "scope"::"key"
  %result, %result_ready = stream.async.parameter.read await(%imm) "scope"::"key"[%c0] -> %target[%c0_index to %c100 for %c100] : !stream.resource<transient>{%target_size} => !stream.timepoint
  util.return %result, %result_ready : !stream.resource<transient>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterWriteWait
util.func private @ElideImmediateAsyncParameterWriteWait(%source: !stream.resource<transient>, %source_size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : i64
  %c0_index = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: await
  // CHECK: stream.async.parameter.write
  %result, %result_ready = stream.async.parameter.write await(%imm) %source[%c0_index to %c100 for %c100] -> "scope"::"key"[%c0] : !stream.resource<transient>{%source_size} => !stream.timepoint
  util.return %result, %result_ready : !stream.resource<transient>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterGatherWait
util.func private @ElideImmediateAsyncParameterGatherWait(%target: !stream.resource<transient>, %target_size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : i64
  %c10 = arith.constant 10 : i64
  %c0_index = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: await
  // CHECK: stream.async.parameter.gather {
  %result, %result_ready = stream.async.parameter.gather await(%imm) {
    "scope"::"key0"[%c0] -> %target[%c0_index to %c100 for %c100] : !stream.resource<transient>{%target_size},
    "scope"::"key1"[%c10] -> %target[%c100 to %c200 for %c100] : !stream.resource<transient>{%target_size}
  } : !stream.resource<transient> => !stream.timepoint
  util.return %result, %result_ready : !stream.resource<transient>, !stream.timepoint
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterScatterWait
util.func private @ElideImmediateAsyncParameterScatterWait(%source: !stream.resource<transient>, %source_size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : i64
  %c10 = arith.constant 10 : i64
  %c0_index = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: await
  // CHECK: stream.async.parameter.scatter {
  %result, %result_ready = stream.async.parameter.scatter await(%imm) {
    %source[%c0_index to %c100 for %c100] : !stream.resource<transient>{%source_size} -> "scope"::"key0"[%c0],
    %source[%c100 to %c200 for %c100] : !stream.resource<transient>{%source_size} -> "scope"::"key1"[%c10]
  } : !stream.resource<transient> => !stream.timepoint
  util.return %result, %result_ready : !stream.resource<transient>, !stream.timepoint
}

// -----

// CHECK-LABEL: @FoldAsyncParameterReadTargetSubviewMultipleReads
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[PARAM_OFFSET1:[a-zA-Z0-9]+]]: i64, %[[PARAM_OFFSET2:[a-zA-Z0-9]+]]: i64, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[READ1_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ1_END:[a-zA-Z0-9]+]]: index, %[[READ1_LENGTH:[a-zA-Z0-9]+]]: index, %[[READ2_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ2_END:[a-zA-Z0-9]+]]: index, %[[READ2_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterReadTargetSubviewMultipleReads(%base: !stream.resource<transient>, %base_size: index, %param_offset1: i64, %param_offset2: i64, %subview_offset: index, %subview_size: index, %read1_offset: index, %read1_end: index, %read1_length: index, %read2_offset: index, %read2_end: index, %read2_length: index) -> (!stream.resource<transient>, !stream.resource<transient>) {
  // Subview used by multiple different reads. Both reads can independently fold.
  // CHECK-NOT: stream.resource.subview
  // First read's arithmetic.
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET1:.+]] = arith.addi %[[PARAM_OFFSET1]], %[[OFFSET1_I64]]
  // CHECK-DAG: %[[ADJUSTED_READ1_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[READ1_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_READ1_END:.+]] = arith.addi %[[ADJUSTED_READ1_OFFSET]], %[[READ1_LENGTH]]
  // First read operation.
  // CHECK: %[[RESULT1:.+]], %[[RESULT1_READY:.+]] = stream.async.parameter.read "scope"::"key1"[%[[ADJUSTED_PARAM_OFFSET1]]] -> %[[BASE]][%[[ADJUSTED_READ1_OFFSET]] to %[[ADJUSTED_READ1_END]] for %[[READ1_LENGTH]]]
  %subview = stream.resource.subview %base[%subview_offset] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview_size}
  %result1, %result1_ready = stream.async.parameter.read "scope"::"key1"[%param_offset1] -> %subview[%read1_offset to %read1_end for %read1_length] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // Second read's arithmetic.
  // CHECK-DAG: %[[OFFSET2_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET2:.+]] = arith.addi %[[PARAM_OFFSET2]], %[[OFFSET2_I64]]
  // CHECK-DAG: %[[ADJUSTED_READ2_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[READ2_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_READ2_END:.+]] = arith.addi %[[ADJUSTED_READ2_OFFSET]], %[[READ2_LENGTH]]
  // Second read operation.
  // CHECK: %[[RESULT2:.+]], %[[RESULT2_READY:.+]] = stream.async.parameter.read "scope"::"key2"[%[[ADJUSTED_PARAM_OFFSET2]]] -> %[[BASE]][%[[ADJUSTED_READ2_OFFSET]] to %[[ADJUSTED_READ2_END]] for %[[READ2_LENGTH]]]
  %result2, %result2_ready = stream.async.parameter.read "scope"::"key2"[%param_offset2] -> %subview[%read2_offset to %read2_end for %read2_length] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT1_SYNC:.+]] = stream.timepoint.await %[[RESULT1_READY]] => %[[RESULT1]]
  %result1_sync = stream.timepoint.await %result1_ready => %result1 : !stream.resource<transient>{%base_size}
  // CHECK: %[[RESULT2_SYNC:.+]] = stream.timepoint.await %[[RESULT2_READY]] => %[[RESULT2]]
  %result2_sync = stream.timepoint.await %result2_ready => %result2 : !stream.resource<transient>{%base_size}
  // CHECK: util.return %[[RESULT1_SYNC]], %[[RESULT2_SYNC]]
  util.return %result1_sync, %result2_sync : !stream.resource<transient>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterWriteSourceSubviewMultipleWrites
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[PARAM_OFFSET1:[a-zA-Z0-9]+]]: i64, %[[PARAM_OFFSET2:[a-zA-Z0-9]+]]: i64, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[WRITE1_OFFSET:[a-zA-Z0-9]+]]: index, %[[WRITE1_END:[a-zA-Z0-9]+]]: index, %[[WRITE1_LENGTH:[a-zA-Z0-9]+]]: index, %[[WRITE2_OFFSET:[a-zA-Z0-9]+]]: index, %[[WRITE2_END:[a-zA-Z0-9]+]]: index, %[[WRITE2_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterWriteSourceSubviewMultipleWrites(%base: !stream.resource<transient>, %base_size: index, %param_offset1: i64, %param_offset2: i64, %subview_offset: index, %subview_size: index, %write1_offset: index, %write1_end: index, %write1_length: index, %write2_offset: index, %write2_end: index, %write2_length: index) -> (!stream.resource<transient>, !stream.resource<transient>) {
  // Subview used by multiple different writes. Both writes can independently fold.
  // CHECK-NOT: stream.resource.subview
  // First write's arithmetic.
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET1:.+]] = arith.addi %[[PARAM_OFFSET1]], %[[OFFSET1_I64]]
  // CHECK-DAG: %[[ADJUSTED_WRITE1_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[WRITE1_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_WRITE1_END:.+]] = arith.addi %[[ADJUSTED_WRITE1_OFFSET]], %[[WRITE1_LENGTH]]
  // First write operation.
  // CHECK: %[[RESULT1:.+]], %[[RESULT1_READY:.+]] = stream.async.parameter.write %[[BASE]][%[[ADJUSTED_WRITE1_OFFSET]] to %[[ADJUSTED_WRITE1_END]] for %[[WRITE1_LENGTH]]] -> "scope"::"key1"[%[[ADJUSTED_PARAM_OFFSET1]]]
  %subview = stream.resource.subview %base[%subview_offset] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview_size}
  %result1, %result1_ready = stream.async.parameter.write %subview[%write1_offset to %write1_end for %write1_length] -> "scope"::"key1"[%param_offset1] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // Second write's arithmetic.
  // CHECK-DAG: %[[OFFSET2_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET2:.+]] = arith.addi %[[PARAM_OFFSET2]], %[[OFFSET2_I64]]
  // CHECK-DAG: %[[ADJUSTED_WRITE2_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[WRITE2_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_WRITE2_END:.+]] = arith.addi %[[ADJUSTED_WRITE2_OFFSET]], %[[WRITE2_LENGTH]]
  // Second write operation.
  // CHECK: %[[RESULT2:.+]], %[[RESULT2_READY:.+]] = stream.async.parameter.write %[[BASE]][%[[ADJUSTED_WRITE2_OFFSET]] to %[[ADJUSTED_WRITE2_END]] for %[[WRITE2_LENGTH]]] -> "scope"::"key2"[%[[ADJUSTED_PARAM_OFFSET2]]]
  %result2, %result2_ready = stream.async.parameter.write %subview[%write2_offset to %write2_end for %write2_length] -> "scope"::"key2"[%param_offset2] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT1_SYNC:.+]] = stream.timepoint.await %[[RESULT1_READY]] => %[[RESULT1]]
  %result1_sync = stream.timepoint.await %result1_ready => %result1 : !stream.resource<transient>{%base_size}
  // CHECK: %[[RESULT2_SYNC:.+]] = stream.timepoint.await %[[RESULT2_READY]] => %[[RESULT2]]
  %result2_sync = stream.timepoint.await %result2_ready => %result2 : !stream.resource<transient>{%base_size}
  // CHECK: util.return %[[RESULT1_SYNC]], %[[RESULT2_SYNC]]
  util.return %result1_sync, %result2_sync : !stream.resource<transient>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterReadTargetNestedSubviews
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[OFFSET1:[a-zA-Z0-9]+]]: index, %[[SUBVIEW1_SIZE:[a-zA-Z0-9]+]]: index, %[[OFFSET2:[a-zA-Z0-9]+]]: index, %[[SUBVIEW2_SIZE:[a-zA-Z0-9]+]]: index, %[[READ_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ_END:[a-zA-Z0-9]+]]: index, %[[READ_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterReadTargetNestedSubviews(%base: !stream.resource<transient>, %base_size: index, %param_offset: i64, %offset1: index, %subview1_size: index, %offset2: index, %subview2_size: index, %read_offset: index, %read_end: index, %read_length: index) -> !stream.resource<transient> {
  // Nested subviews: subview2 = subview(subview1).
  // Both subviews should be completely folded away.
  // CHECK-DAG: %[[COMBINED_OFFSET12:.+]] = arith.addi %[[OFFSET1]], %[[OFFSET2]]
  // CHECK-DAG: %[[COMBINED_OFFSET12_I64:.+]] = arith.index_cast %[[COMBINED_OFFSET12]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET:.+]] = arith.addi %[[PARAM_OFFSET]], %[[COMBINED_OFFSET12_I64]]
  // CHECK-DAG: %[[ADJUSTED_READ_OFFSET:.+]] = arith.addi %[[COMBINED_OFFSET12]], %[[READ_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_READ_END:.+]] = arith.addi %[[ADJUSTED_READ_OFFSET]], %[[READ_LENGTH]]
  // CHECK-NOT: stream.resource.subview
  %subview1 = stream.resource.subview %base[%offset1] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview1_size}
  %subview2 = stream.resource.subview %subview1[%offset2] : !stream.resource<transient>{%subview1_size} -> !stream.resource<transient>{%subview2_size}
  // Both subviews fold completely away.
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read "scope"::"key"[%[[ADJUSTED_PARAM_OFFSET]]] -> %[[BASE]][%[[ADJUSTED_READ_OFFSET]] to %[[ADJUSTED_READ_END]] for %[[READ_LENGTH]]]
  %result, %result_ready = stream.async.parameter.read "scope"::"key"[%param_offset] -> %subview2[%read_offset to %read_end for %read_length] : !stream.resource<transient>{%subview2_size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]] : !stream.resource<transient>{%[[SUBVIEW1_SIZE]]}
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%subview1_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterGatherTargetSubviewMultipleGathers
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[PARAM_OFFSET1:[a-zA-Z0-9]+]]: i64, %[[PARAM_OFFSET2:[a-zA-Z0-9]+]]: i64, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[GATHER1_OFFSET:[a-zA-Z0-9]+]]: index, %[[GATHER1_END:[a-zA-Z0-9]+]]: index, %[[GATHER1_LENGTH:[a-zA-Z0-9]+]]: index, %[[GATHER2_OFFSET:[a-zA-Z0-9]+]]: index, %[[GATHER2_END:[a-zA-Z0-9]+]]: index, %[[GATHER2_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterGatherTargetSubviewMultipleGathers(%base: !stream.resource<transient>, %base_size: index, %param_offset1: i64, %param_offset2: i64, %subview_offset: index, %subview_size: index, %gather1_offset: index, %gather1_end: index, %gather1_length: index, %gather2_offset: index, %gather2_end: index, %gather2_length: index) -> (!stream.resource<transient>, !stream.resource<transient>) {
  // Subview used by multiple different gather operations. Both gathers can independently fold.
  // CHECK-NOT: stream.resource.subview
  // First gather's arithmetic.
  // CHECK-DAG: %[[OFFSET1_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET1:.+]] = arith.addi %[[PARAM_OFFSET1]], %[[OFFSET1_I64]]
  // CHECK-DAG: %[[ADJUSTED_GATHER1_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[GATHER1_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_GATHER1_END:.+]] = arith.addi %[[ADJUSTED_GATHER1_OFFSET]], %[[GATHER1_LENGTH]]
  // First gather operation.
  // CHECK: %[[RESULT1:.+]], %[[RESULT1_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: "scope"::"key1"[%[[ADJUSTED_PARAM_OFFSET1]]] -> %[[BASE]][%[[ADJUSTED_GATHER1_OFFSET]] to %[[ADJUSTED_GATHER1_END]] for %[[GATHER1_LENGTH]]]
  %subview = stream.resource.subview %base[%subview_offset] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview_size}
  %result1, %result1_ready = stream.async.parameter.gather {
    "scope"::"key1"[%param_offset1] -> %subview[%gather1_offset to %gather1_end for %gather1_length] : !stream.resource<transient>{%subview_size}
  } : !stream.resource<transient> => !stream.timepoint
  // Second gather's arithmetic.
  // CHECK-DAG: %[[OFFSET2_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM_OFFSET2:.+]] = arith.addi %[[PARAM_OFFSET2]], %[[OFFSET2_I64]]
  // CHECK-DAG: %[[ADJUSTED_GATHER2_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[GATHER2_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_GATHER2_END:.+]] = arith.addi %[[ADJUSTED_GATHER2_OFFSET]], %[[GATHER2_LENGTH]]
  // Second gather operation.
  // CHECK: %[[RESULT2:.+]], %[[RESULT2_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: "scope"::"key2"[%[[ADJUSTED_PARAM_OFFSET2]]] -> %[[BASE]][%[[ADJUSTED_GATHER2_OFFSET]] to %[[ADJUSTED_GATHER2_END]] for %[[GATHER2_LENGTH]]]
  %result2, %result2_ready = stream.async.parameter.gather {
    "scope"::"key2"[%param_offset2] -> %subview[%gather2_offset to %gather2_end for %gather2_length] : !stream.resource<transient>{%subview_size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT1_SYNC:.+]] = stream.timepoint.await %[[RESULT1_READY]] => %[[RESULT1]]
  %result1_sync = stream.timepoint.await %result1_ready => %result1 : !stream.resource<transient>{%base_size}
  // CHECK: %[[RESULT2_SYNC:.+]] = stream.timepoint.await %[[RESULT2_READY]] => %[[RESULT2]]
  %result2_sync = stream.timepoint.await %result2_ready => %result2 : !stream.resource<transient>{%base_size}
  // CHECK: util.return %[[RESULT1_SYNC]], %[[RESULT2_SYNC]]
  util.return %result1_sync, %result2_sync : !stream.resource<transient>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterScatterSourceSubviewVariadicEntries
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[OFFSET:.+]]: index)
util.func private @FoldAsyncParameterScatterSourceSubviewVariadicEntries(%base: !stream.resource<transient>, %base_size: index, %offset: index) -> !stream.resource<transient> {
  // CHECK-DAG: %[[PARAM_OFFSET1:.+]] = arith.constant 800 : i64
  %param_offset1 = arith.constant 800 : i64
  // CHECK-DAG: %[[PARAM_OFFSET2:.+]] = arith.constant 900 : i64
  %param_offset2 = arith.constant 900 : i64
  // CHECK-DAG: %[[PARAM_OFFSET3:.+]] = arith.constant 1000 : i64
  %param_offset3 = arith.constant 1000 : i64
  // CHECK-DAG: %[[SOURCE_OFFSET1:.+]] = arith.constant 40 : index
  %source_offset1 = arith.constant 40 : index
  %source_end1 = arith.constant 140 : index
  %source_length1 = arith.constant 100 : index
  // CHECK-DAG: %[[SOURCE_OFFSET2:.+]] = arith.constant 50 : index
  %source_offset2 = arith.constant 50 : index
  %source_end2 = arith.constant 150 : index
  %source_length2 = arith.constant 100 : index
  // CHECK-DAG: %[[SOURCE_OFFSET3:.+]] = arith.constant 60 : index
  %source_offset3 = arith.constant 60 : index
  %source_end3 = arith.constant 160 : index
  %source_length3 = arith.constant 100 : index
  %subview_size = arith.constant 800 : index
  // Single scatter with 3 variadic entries using same subview should fold.
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[OFFSET]] : index to i64
  // CHECK-DAG: %[[FOLDED_PARAM1:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_PARAM2:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET2]]
  // CHECK-DAG: %[[FOLDED_PARAM3:.+]] = arith.addi %[[OFFSET_I64]], %[[PARAM_OFFSET3]]
  // CHECK-DAG: %[[FOLDED_SOURCE1:.+]] = arith.addi %[[OFFSET]], %[[SOURCE_OFFSET1]]
  // CHECK-DAG: %[[FOLDED_SOURCE2:.+]] = arith.addi %[[OFFSET]], %[[SOURCE_OFFSET2]]
  // CHECK-DAG: %[[FOLDED_SOURCE3:.+]] = arith.addi %[[OFFSET]], %[[SOURCE_OFFSET3]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %base[%offset] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview_size}
  // All 3 entries should have folded offsets.
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.scatter {
  // CHECK-NEXT: %[[BASE]][%[[FOLDED_SOURCE1]]{{.+}} -> "scope"::"key1"[%[[FOLDED_PARAM1]]]
  // CHECK-NEXT: %[[BASE]][%[[FOLDED_SOURCE2]]{{.+}} -> "scope"::"key2"[%[[FOLDED_PARAM2]]]
  // CHECK-NEXT: %[[BASE]][%[[FOLDED_SOURCE3]]{{.+}} -> "scope"::"key3"[%[[FOLDED_PARAM3]]]
  %result, %result_ready = stream.async.parameter.scatter {
    %subview[%source_offset1 to %source_end1 for %source_length1] : !stream.resource<transient>{%subview_size} -> "scope"::"key1"[%param_offset1],
    %subview[%source_offset2 to %source_end2 for %source_length2] : !stream.resource<transient>{%subview_size} -> "scope"::"key2"[%param_offset2],
    %subview[%source_offset3 to %source_end3 for %source_length3] : !stream.resource<transient>{%subview_size} -> "scope"::"key3"[%param_offset3]
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]]
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%base_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterReadTargetAlignmentOffsets
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[TARGET_SIZE:.+]]: index, %[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[ALIGNMENT_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[READ_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ_END:[a-zA-Z0-9]+]]: index, %[[READ_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterReadTargetAlignmentOffsets(%target: !stream.resource<transient>, %target_size: index, %param_offset: i64, %alignment_offset: index, %subview_size: index, %read_offset: index, %read_end: index, %read_length: index) -> !stream.resource<transient> {
  // Test with common alignment values (4KB, 16KB, 8KB).
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[ALIGNMENT_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM:.+]] = arith.addi %[[PARAM_OFFSET]], %[[OFFSET_I64]]
  // CHECK-DAG: %[[ADJUSTED_READ_OFFSET:.+]] = arith.addi %[[ALIGNMENT_OFFSET]], %[[READ_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_READ_END:.+]] = arith.addi %[[ADJUSTED_READ_OFFSET]], %[[READ_LENGTH]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %target[%alignment_offset] : !stream.resource<transient>{%target_size} -> !stream.resource<transient>{%subview_size}
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read "scope"::"key"[%[[ADJUSTED_PARAM]]] -> %[[TARGET]][%[[ADJUSTED_READ_OFFSET]] to %[[ADJUSTED_READ_END]]
  %result, %result_ready = stream.async.parameter.read "scope"::"key"[%param_offset] -> %subview[%read_offset to %read_end for %read_length] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]]
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%target_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterGatherTargetSubviewMultipleEntries
// CHECK-SAME: (%[[BASE:.+]]: !stream.resource<transient>, %[[BASE_SIZE:.+]]: index, %[[PARAM_OFFSET1:[a-zA-Z0-9]+]]: i64, %[[PARAM_OFFSET2:[a-zA-Z0-9]+]]: i64, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[GATHER1_OFFSET:[a-zA-Z0-9]+]]: index, %[[GATHER1_END:[a-zA-Z0-9]+]]: index, %[[GATHER1_LENGTH:[a-zA-Z0-9]+]]: index, %[[GATHER2_OFFSET:[a-zA-Z0-9]+]]: index, %[[GATHER2_END:[a-zA-Z0-9]+]]: index, %[[GATHER2_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterGatherTargetSubviewMultipleEntries(%base: !stream.resource<transient>, %base_size: index, %param_offset1: i64, %param_offset2: i64, %subview_offset: index, %subview_size: index, %gather1_offset: index, %gather1_end: index, %gather1_length: index, %gather2_offset: index, %gather2_end: index, %gather2_length: index) -> !stream.resource<transient> {
  // Single gather with multiple entries all using same subview target should fold.
  // All entries should have their parameter offsets and target offsets adjusted.
  // First entry's arithmetic.
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM1:.+]] = arith.addi %[[PARAM_OFFSET1]], %[[OFFSET_I64]]
  // CHECK-DAG: %[[ADJUSTED_GATHER1_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[GATHER1_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_GATHER1_END:.+]] = arith.addi %[[ADJUSTED_GATHER1_OFFSET]], %[[GATHER1_LENGTH]]
  // Second entry's arithmetic.
  // CHECK-DAG: %[[ADJUSTED_PARAM2:.+]] = arith.addi %[[PARAM_OFFSET2]], %[[OFFSET_I64]]
  // CHECK-DAG: %[[ADJUSTED_GATHER2_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[GATHER2_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_GATHER2_END:.+]] = arith.addi %[[ADJUSTED_GATHER2_OFFSET]], %[[GATHER2_LENGTH]]
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %base[%subview_offset] : !stream.resource<transient>{%base_size} -> !stream.resource<transient>{%subview_size}
  // Both entries should fold to use base resource with adjusted offsets.
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.gather {
  // CHECK-NEXT: "scope"::"key1"[%[[ADJUSTED_PARAM1]]] -> %[[BASE]][%[[ADJUSTED_GATHER1_OFFSET]] to %[[ADJUSTED_GATHER1_END]] for %[[GATHER1_LENGTH]]] : !stream.resource<transient>{%[[BASE_SIZE]]}
  // CHECK-NEXT: "scope"::"key2"[%[[ADJUSTED_PARAM2]]] -> %[[BASE]][%[[ADJUSTED_GATHER2_OFFSET]] to %[[ADJUSTED_GATHER2_END]] for %[[GATHER2_LENGTH]]] : !stream.resource<transient>{%[[BASE_SIZE]]}
  %result, %result_ready = stream.async.parameter.gather {
    "scope"::"key1"[%param_offset1] -> %subview[%gather1_offset to %gather1_end for %gather1_length] : !stream.resource<transient>{%subview_size},
    "scope"::"key2"[%param_offset2] -> %subview[%gather2_offset to %gather2_end for %gather2_length] : !stream.resource<transient>{%subview_size}
  } : !stream.resource<transient> => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]]
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%base_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @FoldAsyncParameterReadAwaitThenSubviewThenRead
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[TARGET_SIZE:.+]]: index, %[[PARAM_OFFSET1:[a-zA-Z0-9]+]]: i64, %[[PARAM_OFFSET2:[a-zA-Z0-9]+]]: i64, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[READ_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ_END:[a-zA-Z0-9]+]]: index, %[[READ_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @FoldAsyncParameterReadAwaitThenSubviewThenRead(%target: !stream.resource<transient>, %target_size: index, %param_offset1: i64, %param_offset2: i64, %subview_offset: index, %subview_size: index, %read_offset: index, %read_end: index, %read_length: index) -> !stream.resource<transient> {
  // Read → await → subview of awaited resource → read using subview as target.
  // Subview is legal (on awaited resource) and should fold into second read.
  // CHECK: %[[READ1_RESULT:.+]], %[[READ1_READY:.+]] = stream.async.parameter.read "scope"::"key1"[%[[PARAM_OFFSET1]]] -> %[[TARGET]]
  %read1_result, %read1_ready = stream.async.parameter.read "scope"::"key1"[%param_offset1] -> %target[%subview_offset to %read_end for %read_length] : !stream.resource<transient>{%target_size} => !stream.timepoint
  // Arithmetic for second read is hoisted after first read but before await.
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_PARAM:.+]] = arith.addi %[[PARAM_OFFSET2]], %[[OFFSET_I64]]
  // CHECK-DAG: %[[ADJUSTED_READ_OFFSET:.+]] = arith.addi %[[SUBVIEW_OFFSET]], %[[READ_OFFSET]]
  // CHECK-DAG: %[[ADJUSTED_READ_END:.+]] = arith.addi %[[ADJUSTED_READ_OFFSET]], %[[READ_LENGTH]]
  // CHECK: %[[AWAITED:.+]] = stream.timepoint.await %[[READ1_READY]] => %[[READ1_RESULT]]
  %awaited = stream.timepoint.await %read1_ready => %read1_result : !stream.resource<transient>{%target_size}
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %awaited[%subview_offset] : !stream.resource<transient>{%target_size} -> !stream.resource<transient>{%subview_size}
  // Subview should fold into second read.
  // CHECK: %[[RESULT:.+]], %[[RESULT_READY:.+]] = stream.async.parameter.read "scope"::"key2"[%[[ADJUSTED_PARAM]]] -> %[[AWAITED]][%[[ADJUSTED_READ_OFFSET]] to %[[ADJUSTED_READ_END]]
  %result, %result_ready = stream.async.parameter.read "scope"::"key2"[%param_offset2] -> %subview[%read_offset to %read_end for %read_length] : !stream.resource<transient>{%subview_size} => !stream.timepoint
  // CHECK: %[[RESULT_SYNC:.+]] = stream.timepoint.await %[[RESULT_READY]] => %[[RESULT]]
  %result_sync = stream.timepoint.await %result_ready => %result : !stream.resource<transient>{%target_size}
  // CHECK: util.return %[[RESULT_SYNC]]
  util.return %result_sync : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @ElideImmediateAsyncParameterLoadThenSubviewFold
// CHECK-SAME: (%[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[LOAD_SIZE:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index)
util.func private @ElideImmediateAsyncParameterLoadThenSubviewFold(%param_offset: i64, %load_size: index, %subview_offset: index, %subview_size: index) -> !stream.resource<constant> {
  // Test pattern composition: immediate await elision + subview folding.
  // CHECK-NOT: stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-DAG: %[[OFFSET_I64:.+]] = arith.index_cast %[[SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_OFFSET:.+]] = arith.addi %[[PARAM_OFFSET]], %[[OFFSET_I64]]
  // Immediate is elided from await, subview folds into load.
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[ADJUSTED_OFFSET]]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load await(%imm) "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.resource.subview
  %subview = stream.resource.subview %loaded[%subview_offset] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%subview_size}
  // CHECK: %[[LOADED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SUBVIEW_SIZE]]}
  %subview_sync = stream.timepoint.await %loaded_ready => %subview : !stream.resource<constant>{%subview_size}
  // CHECK: util.return %[[LOADED_SYNC]]
  util.return %subview_sync : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @MultipleAsyncParameterFoldsInFunction
// CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<transient>, %[[TARGET_SIZE:.+]]: index, %[[LOAD_PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[LOAD_SIZE:[a-zA-Z0-9]+]]: index, %[[LOAD_SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[LOAD_SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[READ_PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[READ_SUBVIEW_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ_SUBVIEW_SIZE:[a-zA-Z0-9]+]]: index, %[[READ_OFFSET:[a-zA-Z0-9]+]]: index, %[[READ_END:[a-zA-Z0-9]+]]: index, %[[READ_LENGTH:[a-zA-Z0-9]+]]: index)
util.func private @MultipleAsyncParameterFoldsInFunction(%target: !stream.resource<transient>, %target_size: index, %load_param_offset: i64, %load_size: index, %load_subview_offset: index, %load_subview_size: index, %read_param_offset: i64, %read_subview_offset: index, %read_subview_size: index, %read_offset: index, %read_end: index, %read_length: index) -> (!stream.resource<constant>, !stream.resource<transient>) {
  // Test multiple independent folds in same function.
  // First fold: load result subview.
  // CHECK-DAG: %[[LOAD_OFFSET_I64:.+]] = arith.index_cast %[[LOAD_SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[LOAD_ADJUSTED_OFFSET:.+]] = arith.addi %[[LOAD_PARAM_OFFSET]], %[[LOAD_OFFSET_I64]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key1"[%[[LOAD_ADJUSTED_OFFSET]]] : !stream.resource<constant>{%[[LOAD_SUBVIEW_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key1"[%load_param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // Second fold arithmetic appears after load operation.
  // CHECK-DAG: %[[READ_OFFSET_I64:.+]] = arith.index_cast %[[READ_SUBVIEW_OFFSET]] : index to i64
  // CHECK-DAG: %[[READ_ADJUSTED_PARAM:.+]] = arith.addi %[[READ_PARAM_OFFSET]], %[[READ_OFFSET_I64]]
  // CHECK-DAG: %[[READ_ADJUSTED_OFFSET:.+]] = arith.addi %[[READ_SUBVIEW_OFFSET]], %[[READ_OFFSET]]
  // CHECK-DAG: %[[READ_ADJUSTED_END:.+]] = arith.addi %[[READ_ADJUSTED_OFFSET]], %[[READ_LENGTH]]
  // CHECK-NOT: stream.resource.subview %[[LOADED]]
  %load_subview = stream.resource.subview %loaded[%load_subview_offset] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%load_subview_size}
  // Second fold: read target subview (independent of first).
  // CHECK-NOT: stream.resource.subview %[[TARGET]]
  %read_subview = stream.resource.subview %target[%read_subview_offset] : !stream.resource<transient>{%target_size} -> !stream.resource<transient>{%read_subview_size}
  // CHECK: %[[READ_RESULT:.+]], %[[READ_RESULT_READY:.+]] = stream.async.parameter.read "scope"::"key2"[%[[READ_ADJUSTED_PARAM]]] -> %[[TARGET]][%[[READ_ADJUSTED_OFFSET]] to %[[READ_ADJUSTED_END]]
  %read_result, %read_result_ready = stream.async.parameter.read "scope"::"key2"[%read_param_offset] -> %read_subview[%read_offset to %read_end for %read_length] : !stream.resource<transient>{%read_subview_size} => !stream.timepoint
  // CHECK-DAG: %[[READ_RESULT_SYNC:.+]] = stream.timepoint.await %[[READ_RESULT_READY]] => %[[READ_RESULT]]
  %read_result_sync = stream.timepoint.await %read_result_ready => %read_result : !stream.resource<transient>{%target_size}
  // CHECK-DAG: %[[LOADED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[LOAD_SUBVIEW_SIZE]]}
  %load_subview_sync = stream.timepoint.await %loaded_ready => %load_subview : !stream.resource<constant>{%load_subview_size}
  // CHECK: util.return %[[LOADED_SYNC]], %[[READ_RESULT_SYNC]]
  util.return %load_subview_sync, %read_result_sync : !stream.resource<constant>, !stream.resource<transient>
}

// -----

// CHECK-LABEL: @AsyncParameterLoadSliceThenAwaitElision
// CHECK-SAME: (%[[PARAM_OFFSET:[a-zA-Z0-9]+]]: i64, %[[LOAD_SIZE:[a-zA-Z0-9]+]]: index, %[[SLICE_START:[a-zA-Z0-9]+]]: index, %[[SLICE_END:[a-zA-Z0-9]+]]: index, %[[SLICE_SIZE:[a-zA-Z0-9]+]]: index)
util.func private @AsyncParameterLoadSliceThenAwaitElision(%param_offset: i64, %load_size: index, %slice_start: index, %slice_end: index, %slice_size: index) -> !stream.resource<constant> {
  // Load → await → slice → await of same ready timepoint.
  // First await makes resource safe, slice folds, second await uses folded resource.
  // CHECK-DAG: %[[START_I64:.+]] = arith.index_cast %[[SLICE_START]] : index to i64
  // CHECK-DAG: %[[ADJUSTED_OFFSET:.+]] = arith.addi %[[PARAM_OFFSET]], %[[START_I64]]
  // CHECK: %[[LOADED:.+]], %[[LOADED_READY:.+]] = stream.async.parameter.load "scope"::"key"[%[[ADJUSTED_OFFSET]]] : !stream.resource<constant>{%[[SLICE_SIZE]]} => !stream.timepoint
  %loaded, %loaded_ready = stream.async.parameter.load "scope"::"key"[%param_offset] : !stream.resource<constant>{%load_size} => !stream.timepoint
  // CHECK-NOT: stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[LOAD_SIZE]]}
  %awaited = stream.timepoint.await %loaded_ready => %loaded : !stream.resource<constant>{%load_size}
  // CHECK-NOT: stream.async.slice
  %sliced = stream.async.slice %awaited[%slice_start to %slice_end] : !stream.resource<constant>{%load_size} -> !stream.resource<constant>{%slice_size}
  // Second await should use folded resource with folded size.
  // CHECK: %[[SLICED_SYNC:.+]] = stream.timepoint.await %[[LOADED_READY]] => %[[LOADED]] : !stream.resource<constant>{%[[SLICE_SIZE]]}
  %sliced_sync = stream.timepoint.await %loaded_ready => %sliced : !stream.resource<constant>{%slice_size}
  // CHECK: util.return %[[SLICED_SYNC]]
  util.return %sliced_sync : !stream.resource<constant>
}
