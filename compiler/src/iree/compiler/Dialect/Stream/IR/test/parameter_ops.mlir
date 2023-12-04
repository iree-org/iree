// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK: util.global private @parameter_unscoped = #stream.parameter.named<"key"> : tensor<10xf32>
util.global private @parameter_unscoped = #stream.parameter.named<"key"> : tensor<10xf32>
// CHECK: util.global private @parameter_scoped = #stream.parameter.named<"scope"::"key"> : tensor<10xf32>
util.global private @parameter_scoped = #stream.parameter.named<"scope"::"key"> : tensor<10xf32>
// CHECK: util.global private @parameter_config = #stream.parameter.named<"scope"::"key", {some.config = "hello"}> : tensor<10xf32>
util.global private @parameter_config = #stream.parameter.named<"scope"::"key", {some.config = "hello"}> : tensor<10xf32>

// -----

// CHECK-LABEL: @parameterLoad
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint)
func.func @parameterLoad(%wait: !stream.timepoint) -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  // CHECK: = stream.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: "scope"::"key0"[%c50_i64] : !stream.resource<constant>{%c100},
  // CHECK-NEXT: "scope"::"key1"[%c51_i64] : !stream.resource<constant>{%c200}
  // CHECK-NEXT: } => !stream.timepoint
  %results:2, %result_timepoint = stream.parameter.load await(%wait) => {
    "scope"::"key0"[%c50_i64] : !stream.resource<constant>{%c100},
    "scope"::"key1"[%c51_i64] : !stream.resource<constant>{%c200}
  } => !stream.timepoint
  return %results#0, %results#1, %result_timepoint : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterLoadNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint)
func.func @parameterLoadNoScope(%wait: !stream.timepoint) -> (!stream.resource<constant>, !stream.timepoint) {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  // CHECK: = stream.parameter.load await(%[[WAIT]]) => {
  // CHECK-NEXT: "key"[%c50_i64] : !stream.resource<constant>{%c100}
  // CHECK-NEXT: } => !stream.timepoint
  %result, %result_timepoint = stream.parameter.load await(%wait) => {
    "key"[%c50_i64] : !stream.resource<constant>{%c100}
  } => !stream.timepoint
  return %result, %result_timepoint : !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterRead
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>)
func.func @parameterRead(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK: = stream.parameter.read await(%[[WAIT]]) => "scope"::"key"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !stream.resource<transient>{%c300} => !stream.timepoint
  %timepoint = stream.parameter.read await(%wait) => "scope"::"key"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300} => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterWrite
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>)
func.func @parameterWrite(%wait: !stream.timepoint, %source: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK: = stream.parameter.write await(%[[WAIT]]) => %[[SOURCE]][%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key"[%c50_i64] => !stream.timepoint
  %timepoint = stream.parameter.write await(%wait) => %source[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key"[%c50_i64] => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterGather
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>)
func.func @parameterGather(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  %c300 = arith.constant 300 : index
  // CHECK:    = stream.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT:   "scope"::"key0"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !stream.resource<transient>{%c300},
  // CHECK-NEXT:   "scope"::"key1"[%c51_i64] -> %[[TARGET]][%c101 for %c201] : !stream.resource<transient>{%c300},
  // CHECK-NEXT:   "scope"::"key2"[%c52_i64] -> %[[TARGET]][%c102 for %c202] : !stream.resource<transient>{%c300}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.parameter.gather await(%wait) => {
    "scope"::"key0"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300},
    "scope"::"key1"[%c51_i64] -> %target[%c101 for %c201] : !stream.resource<transient>{%c300},
    "scope"::"key2"[%c52_i64] -> %target[%c102 for %c202] : !stream.resource<transient>{%c300}
  } => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterGatherNoScope
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[TARGET:.+]]: !stream.resource<transient>)
func.func @parameterGatherNoScope(%wait: !stream.timepoint, %target: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c300 = arith.constant 300 : index
  // CHECK:    = stream.parameter.gather await(%[[WAIT]]) => {
  // CHECK-NEXT:   "key0"[%c50_i64] -> %[[TARGET]][%c100 for %c200] : !stream.resource<transient>{%c300},
  // CHECK-NEXT:   "key1"[%c51_i64] -> %[[TARGET]][%c101 for %c201] : !stream.resource<transient>{%c300}
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.parameter.gather await(%wait) => {
    "key0"[%c50_i64] -> %target[%c100 for %c200] : !stream.resource<transient>{%c300},
    "key1"[%c51_i64] -> %target[%c101 for %c201] : !stream.resource<transient>{%c300}
  } => !stream.timepoint
  return %timepoint : !stream.timepoint
}

// -----

// CHECK-LABEL: @parameterScatter
// CHECK-SAME: (%[[WAIT:.+]]: !stream.timepoint, %[[SOURCE:.+]]: !stream.resource<transient>)
func.func @parameterScatter(%wait: !stream.timepoint, %source: !stream.resource<transient>) -> !stream.timepoint {
  %c50_i64 = arith.constant 50 : i64
  %c51_i64 = arith.constant 51 : i64
  %c52_i64 = arith.constant 52 : i64
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c102 = arith.constant 102 : index
  %c200 = arith.constant 200 : index
  %c201 = arith.constant 201 : index
  %c202 = arith.constant 202 : index
  %c300 = arith.constant 300 : index
  // CHECK:    = stream.parameter.scatter await(%[[WAIT]]) => {
  // CHECK-NEXT:   %[[SOURCE]][%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key0"[%c50_i64],
  // CHECK-NEXT:   %[[SOURCE]][%c101 for %c201] : !stream.resource<transient>{%c300} -> "scope"::"key1"[%c51_i64],
  // CHECK-NEXT:   %[[SOURCE]][%c102 for %c202] : !stream.resource<transient>{%c300} -> "scope"::"key2"[%c52_i64]
  // CHECK-NEXT: } => !stream.timepoint
  %timepoint = stream.parameter.scatter await(%wait) => {
    %source[%c100 for %c200] : !stream.resource<transient>{%c300} -> "scope"::"key0"[%c50_i64],
    %source[%c101 for %c201] : !stream.resource<transient>{%c300} -> "scope"::"key1"[%c51_i64],
    %source[%c102 for %c202] : !stream.resource<transient>{%c300} -> "scope"::"key2"[%c52_i64]
  } => !stream.timepoint
  return %timepoint : !stream.timepoint
}
