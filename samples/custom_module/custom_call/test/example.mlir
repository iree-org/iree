// RUN: iree-compile %s --iree-execution-model=async-external --iree-hal-target-backends=llvm-cpu | custom-module-async-run - example.main | FileCheck %s

module @example {
  //func.func private @custom_call.Double(tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) attributes {
  func.func private @custom_call.Double(tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32> // attributes {
  //  iree.abi.model = "coarse-fences",
  //  nosideeffects
  //}

  //func.func private @custom_call.Triple(tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) attributes {
  func.func private @custom_call.Triple(tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32> // attributes {
  //  iree.abi.model = "coarse-fences",
  //  nosideeffects
  //}

  func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {

    %c0 = arith.constant 0: index
    %c1 = arith.constant 1: index
    %dim0 = tensor.dim %arg0, %c0: tensor<?x?xi32>
    %dim1 = tensor.dim %arg0, %c1: tensor<?x?xi32>

    // %t0 = arith.muli %arg0, %arg0 : tensor<?x?xi32>
    // %t1 = arith.addi %arg0, %arg0 : tensor<?x?xi32>
    // %i1 = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    // %i2 = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    // %t2:2 = call @Double.function(%t0, %t1, %i1, %i2) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>)
    // %i3 = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    // %i4 = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    // %rt:2 = call @Triple.function(%t2#0, %t2#1, %i3, %i4) : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi32>)
    // return %rt#0 : tensor<?x?xi32>

    %0 = arith.muli %arg0, %arg0 : tensor<?x?xi32>
    %init = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    %p1 = call @custom_call.Double(%init, %dim0, %dim1, %0) : (tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32>
    %init2 = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    %1 = call @custom_call.Triple(%init2, %dim0, %dim1, %p1) : (tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32>
    %res = arith.muli %1, %1 : tensor<?x?xi32>
    return %res : tensor<?x?xi32>
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
