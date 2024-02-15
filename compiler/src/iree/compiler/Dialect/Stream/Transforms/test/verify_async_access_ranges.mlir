// RUN: iree-opt --iree-stream-verify-async-access-ranges --split-input-file %s --verify-diagnostics | FileCheck %s

// Tests that statically-known valid ranges pass verification.

// CHECK: @inRangeCopy
util.func public @inRangeCopy(%source: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %source_size = arith.constant 256 : index
  %target_size = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: = stream.async.copy
  %0 = stream.async.copy %source[%c128 to %c256], %target[%c128 to %c256], %c128 : !stream.resource<*>{%source_size} -> %target as !stream.resource<*>{%target_size}
  util.return %0 : !stream.resource<*>
}

// -----

// Tests that statically-known invalid ranges emit errors.
// For more useful reporting we report all errors on an op so this expects 2.
util.func public @outOfRangeCopy(%source: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %source_size = arith.constant 256 : index
  %target_size = arith.constant 255 : index  // NOTE: too small!
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  // expected-error @+3 {{invalid Read access range [128 to 512 for 128] of resource %arg0 with size 256}}
  // expected-error @+2 {{invalid Write access range [256 to 512 for 128] of resource %arg1 with size 255}}
  // expected-error @+1 {{invalid Write access range [256 to 512 for 128] of resource %0 with size 255}}
  %0 = stream.async.copy %source[%c128 to %c512], %target[%c256 to %c512], %c128 : !stream.resource<*>{%source_size} -> %target as !stream.resource<*>{%target_size}
  util.return %0 : !stream.resource<*>
}

// -----

// Tests that static ranges don't get checked against dynamic sizes.
// In the future we could use data flow analysis to try to bound dynamic values
// and this pass could verify the conditions (size of A < size of B, etc).

// CHECK-LABEL: @dynamicSizes
util.func public @dynamicSizes(%source: !stream.resource<*>, %source_size: index, %target: !stream.resource<*>, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.async.copy
  %0 = stream.async.copy %source[%c0 to %c128], %target[%c0 to %c128], %c128 : !stream.resource<*>{%source_size} -> %target as !stream.resource<*>{%target_size}
  util.return %0 : !stream.resource<*>
}
