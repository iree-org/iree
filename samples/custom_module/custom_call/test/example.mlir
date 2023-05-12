// RUN: iree-compile --iree-execution-model=async-internal --iree-hal-target-backends=llvm-cpu %s | \
// RUN: iree-run-module --module=$IREE_BINARY_DIR/samples/custom_module/custom_call/module$IREE_DYLIB_EXT@create_custom_module --module=- --function=main --input="2x3xi32=[1,2,3,4,5,6]" | \
// RUN: FileCheck %s

module @example {
  func.func private @custom_call.Double(tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32> attributes {iree.abi.model = "sync"}

  func.func private @custom_call.Triple(tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32> attributes {iree.abi.model = "sync"}

  // CHECK-LABEL: EXEC @main
  func.func @main(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {

    %c0 = arith.constant 0: index
    %c1 = arith.constant 1: index
    %dim0 = tensor.dim %arg0, %c0: tensor<?x?xi32>
    %dim1 = tensor.dim %arg0, %c1: tensor<?x?xi32>

    %0 = arith.muli %arg0, %arg0 : tensor<?x?xi32>
    %init = flow.tensor.alloc : tensor<?x?xi32>{%dim0, %dim1}
    %p1 = call @custom_call.Double(%init, %dim0, %dim1, %0) : (tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32>
    %1 = call @custom_call.Triple(%p1, %dim0, %dim1, %p1) : (tensor<?x?xi32>, index, index, tensor<?x?xi32>) -> tensor<?x?xi32>
    %res = arith.muli %1, %1 : tensor<?x?xi32>
    return %res : tensor<?x?xi32>
    // CHECK: 2x3xi32=[36 576 2916][9216 22500 46656]
  }
}
