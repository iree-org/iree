// RUN: iree-compile %s --iree-execution-model=async-external --iree-hal-target-backends=llvm-cpu | custom-module-async-run - example.main | FileCheck %s

module @example {
  //===--------------------------------------------------------------------===//
  // Imports
  //===--------------------------------------------------------------------===//
  // External function declarations for the methods implemented in the custom
  // module C++ file. Note that they are prefixed with the `custom.` module
  // name.

  // Asynchronous call that takes/returns a tensor.
  // IREE will pass in a HAL fence indicating when the input tensor is available
  // and a HAL fence that the call can use to indicate when the returned tensor
  // is available. It's expected that the call will not block.
  //
  // Note that `nosideeffects` is critical to ensuring asynchronous execution.
  // When omitted IREE will still pass in the fences but wait on the signal
  // fence after the call completes before continuing. This may be required when
  // returning custom types or synchronizing with external systems.
  func.func private @custom.call.async(tensor<?xi32>) -> tensor<?xi32> attributes {
    iree.abi.model = "coarse-fences",
    nosideeffects
  }

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//
  // Note that there can be any number of publicly-exported methods; this simple
  // sample just has one to keep things simple.

  func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // Compiler-generated dispatch work to show dataflow.
    %0 = arith.muli %arg0, %arg0 : tensor<?xi32>

    // Custom call to an asynchronous import.
    // The runtime will chain together the async work to produce %0 and make the
    // call with a wait fence indicating when %0 is ready. The call *should*
    // return immediately with a newly allocated but not yet populated %1. The
    // runtime will then continue to chain the subsequent %2 work pending the
    // signal from the call indicating that %1 is ready for use.
    //
    // Note that allocations are generally blocking unless performed with the
    // queue-ordered allocation APIs that chain on to fences.
    %1 = call @custom.call.async(%0) : (tensor<?xi32>) -> tensor<?xi32>

    // More generated dispatch work to show dataflow.
    %2 = arith.muli %1, %1 : tensor<?xi32>

    return %2 : tensor<?xi32>
  }

  // TODO(benvanik): fix wait-before-signal on queue-ordered allocations.
  // For now we have to signal to T=1 before invoking the function but that's
  // only temporary.
  // CHECK: INITIALIZE T=0
  // CHECK: SIGNALED T=1
  // CHECK: VM INVOKE BEGIN example.main
  // CHECK: VM INVOKE END
  // CHECK: REACHED T=2
  // CHECK: MATCHED!
}
