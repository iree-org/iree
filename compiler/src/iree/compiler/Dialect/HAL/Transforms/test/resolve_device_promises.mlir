// RUN: iree-opt --split-input-file --iree-hal-resolve-device-promises %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// Resolves device promises.

// CHECK: module @module
module @module attributes {
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@device0, [1, 2, 3]>
  stream.affinity = #hal.device.promise<@device0, [1, 2, 3]>
} {
  util.global private @device0 = #hal.device.target<"vmvx"> : !hal.device
  util.global private @device1 = #hal.device.target<"vmvx"> : !hal.device
  // CHECK: util.func private @func
  util.func private @func(%arg0: tensor<i32> {
    // CHECK-SAME: arg.affinity = #hal.device.affinity<@device1>
    arg.affinity = #hal.device.promise<@device1>
  }) -> (tensor<i32> {
    // CHECK-SAME: result.affinity = #hal.device.affinity<@device1>
    result.affinity = #hal.device.promise<@device1>
  }) attributes {
    // CHECK-SAME: func.affinity = #hal.device.affinity<@device1>
    func.affinity = #hal.device.promise<@device1>
  } {
    // CHECK: util.return
    util.return {
      // CHECK-SAME: some.affinities = [#hal.device.affinity<@device0>, #hal.device.affinity<@device1>]
      some.affinities = [#hal.device.promise<@device0>, #hal.device.promise<@device1>]
    } %arg0 : tensor<i32>
  }
}

// -----

// Verifies that promised devices exist.

module @module {
  util.global private @device = #hal.device.target<"vmvx"> : !hal.device
  // expected-error@+1 {{op references a promised device that was not declared}}
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.promise<@unknown_device>
  } {
    util.return
  }
}
