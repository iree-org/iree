// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-materialize-encode-ops)' %s | FileCheck %s

#encoding = #iree_encoding.testing_encoding<>
util.func public @fold_tensor_encode_op(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-LABEL: @fold_tensor_encode_op
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9]+]]
// CHECK-NOT:     stream.tensor.encode
// CHECK:         return %[[ARG]]

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @encode_static_shape(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<4x5xf32> in !stream.resource<*>{%arg1}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups()
// CHECK-NEXT:      flow.dispatch.workgroup_count_from_slice
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<readonly:tensor<4x5xf32>>
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<writeonly:tensor<4x5xf32, #[[ENCODING]]>>
// CHECK:           %[[VAL:.+]] = flow.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<4x5xf32> -> tensor<4x5xf32, #[[ENCODING]]>
// CHECK:           flow.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
// CHECK-LABEL:   util.func public @encode_static_shape(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @mixed_static_dynamic_encoding(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<4x?xf32>{%arg2} in !stream.resource<*>{%arg1}
    -> tensor<?x5xf32, #encoding>{%arg3} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
// CHECK-NEXT:      flow.dispatch.workgroup_count_from_slice %[[ARG0]], %[[ARG1]]
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D1:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D1_ARG]], 0 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D0_ARG]], 1 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<readonly:tensor<4x?xf32>>{%[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<writeonly:tensor<?x5xf32, #[[ENCODING]]>>{%[[DEST_D0]]}
// CHECK:           %[[VAL:.+]] = flow.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<4x?xf32> -> tensor<?x5xf32, #[[ENCODING]]>
// CHECK:           flow.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], 5], strides = [1, 1]
// CHECK-LABEL:   util.func public @mixed_static_dynamic_encoding(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @encode_result_resource(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      flow.dispatch.workgroup_count_from_slice %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING]]>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = flow.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
// CHECK:           flow.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @encode_result_resource(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}

// -----

#encoding = #iree_encoding.testing_encoding<>
util.func public @decode_source_resource(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      flow.dispatch.workgroup_count_from_slice %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING]]>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = flow.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[DECODED_VAL:.+]] =  iree_encoding.unset_encoding %[[VAL]] : tensor<?x?xf32, #[[ENCODING]]> -> tensor<?x?xf32>
// CHECK:           flow.dispatch.tensor.store %[[DECODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @decode_source_resource(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}

// -----

#encoding0 = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
#encoding1 = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<456>]>
util.func public @update_encoding(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32, #encoding0>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding1>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING0:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-DAG:  #[[ENCODING1:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<456>]>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      flow.dispatch.workgroup_count_from_slice %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  flow.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  flow.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING0]]>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING1]]>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = flow.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[DECODED_VAL:.+]] = iree_encoding.unset_encoding %[[VAL]] : tensor<?x?xf32, #[[ENCODING0]]> -> tensor<?x?xf32>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[ENCODED_VAL:.+]] = iree_encoding.set_encoding %[[DECODED_VAL]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING1]]>
// CHECK:           flow.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @update_encoding(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}

// -----

// This tests that only a single executable is created and it is reused by both
// dispatch ops.

#encoding = #iree_encoding.testing_encoding<>
util.func public @multi_identical_encode_ops(%arg0: !stream.resource<*>, %arg1: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<4x5xf32> in !stream.resource<*>{%arg1}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%arg1}
  %1 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<4x5xf32> in !stream.resource<*>{%arg1}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%arg1}
  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}
// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups()
// CHECK-NOT:  stream.executable
// CHECK-LABEL:   util.func public @multi_identical_encode_ops(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK:           stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
// CHECK-SAME:        (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
// CHECK-SAME:        -> !stream.resource<*>{%[[TOTAL_SIZE]]}
