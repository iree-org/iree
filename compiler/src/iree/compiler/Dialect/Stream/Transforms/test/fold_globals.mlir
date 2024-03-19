// RUN: iree-opt --split-input-file --iree-util-fold-globals %s | FileCheck %s

// NOTE: this file is only testing that the iree-util-fold-globals pass works
// with stream types - the rest of the testing for that pass lives in
// iree/compiler/Dialect/Util/Transforms/test/fold_globals.mlir

// CHECK: util.global public mutable @uniformConstants = #stream.timepoint<immediate>
util.global public mutable @uniformConstants : !stream.timepoint
util.func public @foo() {
  %timepoint = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: util.global.store
  util.global.store %timepoint, @uniformConstants : !stream.timepoint
  util.return
}
util.func public @bar() {
  %timepoint = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NOT: util.global.store
  util.global.store %timepoint, @uniformConstants : !stream.timepoint
  util.return
}

// -----

// CHECK-NOT: @immutable
util.global private @immutable = #stream.timepoint<immediate> : !stream.timepoint
util.func public @foo() -> !stream.timepoint {
  // CHECK-NOT: util.global.load @immutable
  // CHECK: %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint
  %0 = util.global.load @immutable : !stream.timepoint
  // CHECK: util.return %[[IMMEDIATE]]
  util.return %0 : !stream.timepoint
}
