// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu | custom-module-sync-run - example.main | FileCheck %s

module @example {
  //===--------------------------------------------------------------------===//
  // Imports
  //===--------------------------------------------------------------------===//
  // External function declarations for the methods implemented in the custom
  // module C++ file. Note that they are prefixed with the `custom.` module
  // name.

  // Synchronous call that takes/returns a tensor.
  // IREE will block and wait until the input tensor is available, make the
  // import call, and assume that the returned tensor is immediately available
  // for use.
  func.func private @custom.call.sync(tensor<?xi32>) -> tensor<?xi32>

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//
  // Note that there can be any number of publicly-exported methods; this simple
  // sample just has one to keep things simple.

  // CHECK-LABEL: INVOKE BEGIN example.main
  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // Compiler-generated dispatch work to show dataflow.
    %0 = arith.muli %arg0, %arg0 : tensor<?xi32>

    // Custom call to a synchronous import.
    // The runtime will block and wait until %0 is ready before making the call
    // and assume it can immediately start using the resulting %1 after the call
    // returns. Note that the top-level invocation will block while this call is
    // made and if we were running the compiler-generated dispatches above/below
    // on a GPU it would fully synchronize the host and device (really bad!).
    %1 = call @custom.call.sync(%0) : (tensor<?xi32>) -> tensor<?xi32>

    // More generated dispatch work to show dataflow.
    %2 = arith.muli %1, %1 : tensor<?xi32>

    // CHECK: MATCHED!
    return %2 : tensor<?xi32>
  }
  // CHECK-NEXT: INVOKE END
}
