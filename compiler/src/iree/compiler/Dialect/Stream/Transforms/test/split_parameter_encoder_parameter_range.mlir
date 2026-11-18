// RUN: iree-opt --split-input-file --iree-stream-split-parameter-encoder='mode=consolidate output-scope=encoded' %s | FileCheck %s

// A parameter can refer to a byte range of an archive entry through the
// `offset` and `length` entries of its config. The encoder reads exactly that
// range: the unmodified copy and the input of the hoisted fill both load 1024
// bytes at offset 4096 of "weight", not the start of the entry.

// CHECK-LABEL: module @encoder
// CHECK-LABEL: util.func public @__encode_parameters_all(
// CHECK-DAG: %[[LENGTH:.+]] = arith.constant 1024 : index
// CHECK-DAG: %[[MODEL:.+]] = util.buffer.constant : !util.buffer = "model"
// CHECK-DAG: %[[KEY:.+]] = util.buffer.constant : !util.buffer = "weight"
// CHECK-DAG: %[[OFFSET:.+]] = arith.constant 4096 : i64
// CHECK: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[LENGTH]]}
// CHECK: stream.async.fill
// CHECK: stream.async.parameter.scatter
// CHECK: stream.async.parameter.load {{.*}} %[[MODEL]]::%[[KEY]][%[[OFFSET]]] : !stream.resource<constant>{%[[LENGTH]]}
// CHECK: stream.async.parameter.scatter
// CHECK: util.return

// The main module reads only the encoded archive.
// CHECK-LABEL: module @parameter_range
// CHECK-NOT: #stream.parameter.named<"model"::
module @parameter_range {
  util.global private @weight : !stream.resource<constant>
  util.global private @filled_weight : !stream.resource<constant>
  util.initializer {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c42_i32 = arith.constant 42 : i32
    %param = stream.async.constant : !stream.resource<constant>{%c1024} =
      #stream.parameter.named<"model"::"weight", {offset = 4096 : i64, length = 1024 : i64}> : vector<1024xi8>
    %transferred = stream.async.transfer %param :
      !stream.resource<constant>{%c1024} -> !stream.resource<constant>{%c1024}
    util.global.store %transferred, @weight : !stream.resource<constant>
    %filled = stream.async.fill %c42_i32, %param[%c0 to %c256 for %c256] :
      i32 -> %param as !stream.resource<constant>{%c1024}
    util.global.store %filled, @filled_weight : !stream.resource<constant>
    util.return
  }
}
