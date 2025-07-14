// RUN: iree-opt --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bitcast-unsupported-element-types, canonicalize))" \
// RUN:   --allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s

util.func private @f4_input_tensor(%arg0: tensor<1024xf4E2M1FN>, %arg1: tensor<?x1024xf4E2M1FN>, %arg2: index) -> tensor<1024xi8> {
  %0 = flow.dispatch.workgroups[%arg2](%arg0, %arg1, %arg2)
    : (tensor<1024xf4E2M1FN>, tensor<?x1024xf4E2M1FN>{%arg2}, index) -> tensor<1024xi8> = (
        %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf4E2M1FN>>,
        %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024xf4E2M1FN>>,
        %arg5: index,
        %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xi8>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [1024], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024xf4E2M1FN>> -> tensor<1024xf4E2M1FN>
    %2 = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [%arg5, 1024], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x1024xf4E2M1FN>>{%arg5} -> tensor<?x1024xf4E2M1FN>
    %3 = "dispatch.body"(%1, %2) : (tensor<1024xf4E2M1FN>, tensor<?x1024xf4E2M1FN>) -> (tensor<1024xi8>)
    iree_tensor_ext.dispatch.tensor.store %3, %arg6, offsets = [0], sizes = [1024], strides = [1]
      : tensor<1024xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024xi8>>
    flow.return
  }
  util.return %0 : tensor<1024xi8>
}

// CHECK-LABEL: @f4_input_tensor

// CHECK-SAME:    %[[ARG0:[A-Za-z0-9]+]]: tensor<1024xf4E2M1FN>
// CHECK-SAME:    %[[ARG1:[A-Za-z0-9]+]]: tensor<?x1024xf4E2M1FN>
// CHECK-SAME:    %[[ARG2:[A-Za-z0-9]+]]: index

// CHECK-DAG:     %[[B0:.+]] = flow.tensor.bitcast %[[ARG0]] : tensor<1024xf4E2M1FN> -> tensor<512xi8>
// CHECK-DAG:     %[[B1:.+]] = flow.tensor.bitcast %[[ARG1]] : tensor<?x1024xf4E2M1FN>{%[[ARG2]]} -> tensor<?x512xi8>{%[[ARG2]]}

// CHECK:         flow.dispatch.workgroups[%[[ARG2]]](%[[B0]], %[[B1]], %[[ARG2]])
// CHECK-NEXT:      (%[[ARG3:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xi8>>
// CHECK-SAME:       %[[ARG4:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x512xi8>>
// CHECK-SAME:       %[[ARG5:[A-Za-z0-9]+]]: index

// CHECK-DAG:       %[[L0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG3]], offsets = [0], sizes = [512]
// CHECK-DAG:       %[[B2:.+]] = iree_tensor_ext.bitcast %[[L0]] : tensor<512xi8> -> tensor<1024xf4E2M1FN>
// CHECK-DAG:       %[[L1:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG4]], offsets = [0, 0], sizes = [%[[ARG5]], 512]
// CHECK-DAG:       %[[B3:.+]] = iree_tensor_ext.bitcast %[[L1]] : tensor<?x512xi8>{%[[ARG5]]} -> tensor<?x1024xf4E2M1FN>{%[[ARG5]]}

// CHECK:           "dispatch.body"(%[[B2]], %[[B3]])

// -----

util.func private @f6_input_tensor(%arg0: tensor<960xf6E3M2FN>) -> tensor<960xi8> {
  %0 = flow.dispatch.workgroups(%arg0)
    : (tensor<960xf6E3M2FN>) -> tensor<960xi8> = (
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [960], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>> -> tensor<960xf6E3M2FN>
    %2 = "dispatch.body"(%1) : (tensor<960xf6E3M2FN>) -> (tensor<960xi8>)
    iree_tensor_ext.dispatch.tensor.store %2, %arg2, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
    flow.return
  }
  util.return %0 : tensor<960xi8>
}

// CHECK-LABEL: @f6_input_tensor
// CHECK-SAME:    %[[ARG0:[A-Za-z0-9]+]]: tensor<960xf6E3M2FN>
// CHECK:         %[[B0:.+]] = flow.tensor.bitcast %[[ARG0]] : tensor<960xf6E3M2FN> -> tensor<720xi8>
// CHECK:         flow.dispatch.workgroups(%[[B0]])
// CHECK-NEXT:      %[[ARG1:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<720xi8>>
// CHECK-DAG:       %[[L0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG1]], offsets = [0], sizes = [720]
// CHECK-DAG:       %[[B1:.+]] = iree_tensor_ext.bitcast %[[L0]] : tensor<720xi8> -> tensor<960xf6E3M2FN>
// CHECK:           "dispatch.body"(%[[B1]])

// -----

util.func private @unsupported_dynamic_cast(%arg0: tensor<?xf6E3M2FN>, %arg1: index) -> tensor<960xi8> {
  // expected-error@+1 {{Unsupported tensor type unable to pack to bytes.}}
  %0 = flow.dispatch.workgroups[%arg1](%arg0, %arg1)
    : (tensor<?xf6E3M2FN>{%arg1}, index) -> tensor<960xi8> = (
        %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf6E3M2FN>>,
        %arg3: index,
        %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [%arg3], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf6E3M2FN>>{%arg3} -> tensor<?xf6E3M2FN>
    %2 = "dispatch.body"(%1) : (tensor<?xf6E3M2FN>) -> (tensor<960xi8>)
    iree_tensor_ext.dispatch.tensor.store %2, %arg4, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
    flow.return
  }
  util.return %0 : tensor<960xi8>
}

// -----

util.func private @f6_output_tensor(%arg0: tensor<960xi8>) -> tensor<960xf6E3M2FN> {
  %0 = flow.dispatch.workgroups(%arg0)
    : (tensor<960xi8>) -> tensor<960xf6E3M2FN> = (
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xi8>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xf6E3M2FN>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [960], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xi8>> -> tensor<960xi8>
    %2 = "dispatch.body"(%1) : (tensor<960xi8>) -> (tensor<960xf6E3M2FN>)
    iree_tensor_ext.dispatch.tensor.store %2, %arg2, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xf6E3M2FN> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xf6E3M2FN>>
    flow.return
  }
  util.return %0 : tensor<960xf6E3M2FN>
}

// CHECK-LABEL: @f6_output_tensor
// CHECK:         %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-NEXT:      %[[ARG2:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<720xi8>>
// CHECK:           %[[BODY:.+]] = "dispatch.body"
// CHECK:           %[[B0:.+]] = iree_tensor_ext.bitcast %[[BODY]] : tensor<960xf6E3M2FN> -> tensor<720xi8>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[B0]], %[[ARG2]], offsets = [0], sizes = [720]
// CHECK:         %[[B1:.+]] = flow.tensor.bitcast %[[DISPATCH]] : tensor<720xi8> -> tensor<960xf6E3M2FN>

// -----

util.func private @bitcast_chain_tied(%arg0: tensor<960xf6E3M2FN>) -> tensor<960xf6E3M2FN> {
  %0 = flow.dispatch.workgroups(%arg0)
    : (tensor<960xf6E3M2FN>) -> tensor<960xf6E3M2FN> = (
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xf6E3M2FN>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [960], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>> -> tensor<960xf6E3M2FN>
    %2 = "dispatch0.body"(%1) : (tensor<960xf6E3M2FN>) -> (tensor<960xf6E3M2FN>)
    iree_tensor_ext.dispatch.tensor.store %2, %arg2, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xf6E3M2FN> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xf6E3M2FN>>
    flow.return
  }
  %3 = flow.dispatch.workgroups(%0) : (tensor<960xf6E3M2FN>) -> %0 = (
        %arg3: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<960xf6E3M2FN>>
      ) {
    %4 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0], sizes = [960], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<960xf6E3M2FN>> -> tensor<960xf6E3M2FN>
    %5 = "dispatch1.body"(%4) : (tensor<960xf6E3M2FN>) -> (tensor<960xf6E3M2FN>)
    iree_tensor_ext.dispatch.tensor.store %5, %arg3, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xf6E3M2FN> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<960xf6E3M2FN>>
    flow.return
  }
  util.return %3 : tensor<960xf6E3M2FN>
}

// CHECK-LABEL: @bitcast_chain_tied
// CHECK-SAME:    %[[ARG0:[A-Za-z0-9]+]]: tensor<960xf6E3M2FN>
// CHECK:         %[[B0:.+]] = flow.tensor.bitcast %[[ARG0]] : tensor<960xf6E3M2FN> -> tensor<720xi8>
// CHECK:         %[[D0:.+]] = flow.dispatch.workgroups(%[[B0]])
// CHECK-NEXT:      %[[ARG1:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<720xi8>>
// CHECK-SAME:      %[[ARG2:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<720xi8>>
// CHECK-DAG:       %[[L0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG1]], offsets = [0], sizes = [720]
// CHECK-DAG:       %[[B1:.+]] = iree_tensor_ext.bitcast %[[L0]] : tensor<720xi8> -> tensor<960xf6E3M2FN>
// CHECK:           %[[BODY0:.+]] = "dispatch0.body"(%[[B1]])
// CHECK:           %[[B2:.+]] = iree_tensor_ext.bitcast %[[BODY0]] : tensor<960xf6E3M2FN> -> tensor<720xi8>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[B2]], %[[ARG2]], offsets = [0], sizes = [720]
// CHECK:         %[[D1:.+]] = flow.dispatch.workgroups(%[[D0]])
// CHECK-NEXT:      %[[ARG3:[A-Za-z0-9]+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<720xi8>>
// CHECK:           %[[L1:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG3]], offsets = [0], sizes = [720]
// CHECK:           %[[B3:.+]] = iree_tensor_ext.bitcast %[[L1]] : tensor<720xi8> -> tensor<960xf6E3M2FN>
// CHECK:           %[[BODY1:.+]] = "dispatch1.body"(%[[B3]])
// CHECK:           %[[B4:.+]] = iree_tensor_ext.bitcast %[[BODY1]] : tensor<960xf6E3M2FN> -> tensor<720xi8>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[B4]], %[[ARG3]], offsets = [0], sizes = [720]
// CHECK:         %[[B5:.+]] = flow.tensor.bitcast %[[D1]] : tensor<720xi8> -> tensor<960xf6E3M2FN>
// CHECK:         util.return %[[B5]]

// -----

util.func private @unsupported_other_user(%arg0: tensor<960xf6E3M2FN>) -> tensor<960xi8> {
  // expected-error@+1 {{non-tensor load or store user of unsupported element type.}}
  %0 = flow.dispatch.workgroups(%arg0)
    : (tensor<960xf6E3M2FN>) -> tensor<960xi8> = (
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
      ) {
    %1 = "unsupported.other_user"(%arg1) : (!iree_tensor_ext.dispatch.tensor<readonly:tensor<960xf6E3M2FN>>) -> (tensor<960xi8>)
    iree_tensor_ext.dispatch.tensor.store %1, %arg2, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xi8>>
    flow.return
  }
  util.return %0 : tensor<960xi8>
}

// -----

util.func private @unimplemented_complex(%arg0: tensor<960xcomplex<f32>>) -> tensor<960xcomplex<f32>> {
  %0 = flow.dispatch.workgroups(%arg0)
    : (tensor<960xcomplex<f32>>) -> tensor<960xcomplex<f32>> = (
        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xcomplex<f32>>>,
        %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xcomplex<f32>>>
      ) {
    %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [960], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<960xcomplex<f32>>> -> tensor<960xcomplex<f32>>
    %2 = "dispatch0.body"(%1) : (tensor<960xcomplex<f32>>) -> (tensor<960xcomplex<f32>>)
    iree_tensor_ext.dispatch.tensor.store %2, %arg2, offsets = [0], sizes = [960], strides = [1]
      : tensor<960xcomplex<f32>> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<960xcomplex<f32>>>
    flow.return
  }
  util.return %0 : tensor<960xcomplex<f32>>
}

// CHECK-LABEL: @unimplemented_complex
// CHECK-SAME:    %[[ARG0:[A-Za-z0-9]+]]: tensor<960xcomplex<f32>>
// CHECK:         %[[D0:.+]] = flow.dispatch.workgroups(%[[ARG0]])
// CHECK:         util.return %[[D0]]
