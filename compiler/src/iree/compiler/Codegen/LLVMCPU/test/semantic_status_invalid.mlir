// RUN: iree-opt --iree-llvmcpu-propagate-dispatch-status --iree-convert-to-llvm --split-input-file --verify-diagnostics %s

module {
  // expected-error @+1 {{dispatch entry point must have type () -> ()}}
  func.func @entry(%status: i32) {
    util.status.check_ok %status
    return
  }
}

// -----

module {
  // expected-error @+1 {{semantic failure requires direct calls within the executable}}
  func.func private @helper(%status: i32) {
    util.status.check_ok %status
    return
  }
  func.func @entry() {
    %ptr = func.constant @helper : (i32) -> ()
    %status = arith.constant 9 : i32
    func.call_indirect %ptr(%status) : (i32) -> ()
    return
  }
}
