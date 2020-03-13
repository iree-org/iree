// Tests printing and parsing of dispatch region ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @singleArg
func @singleArg(%arg0 : tensor<?xf32>) {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = "some.shape"
  // CHECK-NEXT: flow.dispatch.region[%[[WORKLOAD]] : index](%arg1 = %arg0 : tensor<?xf32>) {
  // CHECK-NEXT:   flow.return
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> index
  flow.dispatch.region[%workload : index](%i0 = %arg0 : tensor<?xf32>) {
    flow.return
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @multipleArgs
func @multipleArgs(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = "some.shape"
  // CHECK-NEXT: flow.dispatch.region[%[[WORKLOAD]] : index](%arg2 = %arg0 : tensor<?xf32>, %arg3 = %arg1 : tensor<?xf32>) {
  // CHECK-NEXT:   flow.return
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> index
  flow.dispatch.region[%workload : index](%i0 = %arg0 : tensor<?xf32>, %i1 = %arg1 : tensor<?xf32>) {
    flow.return
  }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: @singleResult
func @singleResult(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = "some.shape"
  // CHECK-NEXT: %1 = flow.dispatch.region[%[[WORKLOAD]] : index](%arg1 = %arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT:   flow.return %arg1 : tensor<?xf32>
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> index
  %ret0 = flow.dispatch.region[%workload : index](%i0 = %arg0 : tensor<?xf32>) -> tensor<?xf32> {
    flow.return %i0 : tensor<?xf32>
  }
  // CHECK-NEXT: return %1 : tensor<?xf32>
  return %ret0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @multipleResults
func @multipleResults(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK-NEXT: %[[WORKLOAD:.+]] = "some.shape"
  // CHECK-NEXT: %1:2 = flow.dispatch.region[%[[WORKLOAD]] : index](%arg1 = %arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK-NEXT:   flow.return %arg1, %arg1 : tensor<?xf32>, tensor<?xf32>
  // CHECK-NEXT: }
  %workload = "some.shape"(%arg0) : (tensor<?xf32>) -> index
  %ret0, %ret1 = flow.dispatch.region[%workload : index](%i0 = %arg0 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
    flow.return %i0, %i0 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK-NEXT: return %1#0, %1#1 : tensor<?xf32>, tensor<?xf32>
  return %ret0, %ret1 : tensor<?xf32>, tensor<?xf32>
}
