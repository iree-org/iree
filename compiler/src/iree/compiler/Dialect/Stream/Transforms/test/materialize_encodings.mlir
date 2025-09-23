// RUN: iree-opt --iree-stream-materialize-encodings --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.testing<>
// CHECK-LABEL: @fold_tensor_encode_op
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func public @fold_tensor_encode_op(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  // CHECK-NOT: stream.tensor.encode
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  // CHECK: return %[[ARG0]]
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups()
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice()
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5xf32>>
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5xf32, #[[ENCODING]]>>
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<4x5xf32> -> tensor<4x5xf32, #[[ENCODING]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
// CHECK-LABEL:   util.func public @encode_static_shape(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
util.func public @encode_static_shape(%resource: !stream.resource<*>, %total_size: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<4x5xf32> in !stream.resource<*>{%total_size}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%total_size}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index)
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]])
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D1_ARG]], 0 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D0_ARG]], 1 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xf32>>{%[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x5xf32, #[[ENCODING]]>>{%[[DEST_D0]]}
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [4, %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<4x?xf32> -> tensor<?x5xf32, #[[ENCODING]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], 5], strides = [1, 1]
// CHECK-LABEL:   util.func public @mixed_static_dynamic_encoding(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[SRC_D1:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[DEST_D0:[a-zA-Z0-9]+]]
util.func public @mixed_static_dynamic_encoding(%resource: !stream.resource<*>, %total_size: index, %src_d1: index, %dest_d0: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[SRC_D1]], %[[DEST_D0]]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}], %[[SRC_D1]], %[[DEST_D0]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<4x?xf32>{%src_d1} in !stream.resource<*>{%total_size}
    -> tensor<?x5xf32, #encoding>{%dest_d0} in !stream.resource<*>{%total_size}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING]]>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[ENCODED_VAL:.+]] =  iree_encoding.set_encoding %[[VAL]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @encode_result_resource(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
util.func public @encode_result_resource(%resource: !stream.resource<*>, %total_size: index, %d0: index, %d1: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<?x?xf32>{%d0, %d1} in !stream.resource<*>{%total_size}
    -> tensor<?x?xf32, #encoding>{%d0, %d1} in !stream.resource<*>{%total_size}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING]]>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[DECODED_VAL:.+]] =  iree_encoding.unset_encoding %[[VAL]] : tensor<?x?xf32, #[[ENCODING]]> -> tensor<?x?xf32>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[DECODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @decode_source_resource(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
util.func public @decode_source_resource(%resource: !stream.resource<*>, %total_size: index, %d0: index, %d1: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<?x?xf32, #encoding>{%d0, %d1} in !stream.resource<*>{%total_size}
    -> tensor<?x?xf32>{%d0, %d1} in !stream.resource<*>{%total_size}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-DAG:  #[[ENCODING0:.+]] = #iree_encoding.testing<[#iree_encoding.specialized<123>]>
// CHECK-DAG:  #[[ENCODING1:.+]] = #iree_encoding.testing<[#iree_encoding.specialized<456>]>
#encoding0 = #iree_encoding.testing<[#iree_encoding.specialized<123>]>
#encoding1 = #iree_encoding.testing<[#iree_encoding.specialized<456>]>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]])
// CHECK:         func.func @[[$ENTRY]](
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[SRC_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[SRC_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D0_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_D1_ARG:[a-zA-Z0-9]+]]: index
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-DAG:       %[[SRC_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D0_ARG]], 0 : index
// CHECK-DAG:       %[[SRC_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[SRC_D1_ARG]], 1 : index
// CHECK-DAG:       %[[DEST_D0:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D0_ARG]], 2 : index
// CHECK-DAG:       %[[DEST_D1:.+]] =  iree_tensor_ext.dispatch.workload.ordinal %[[DEST_D1_ARG]], 3 : index
// CHECK:           %[[SRC_BUF:.+]] =  stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING0]]>>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[DEST_BUF:.+]] =  stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING1]]>>{%[[DEST_D0]], %[[DEST_D1]]}
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[SRC_D0]], %[[SRC_D1]]], strides = [1, 1]
// CHECK:           %[[DECODED_VAL:.+]] = iree_encoding.unset_encoding %[[VAL]] : tensor<?x?xf32, #[[ENCODING0]]> -> tensor<?x?xf32>{%[[SRC_D0]], %[[SRC_D1]]}
// CHECK:           %[[ENCODED_VAL:.+]] = iree_encoding.set_encoding %[[DECODED_VAL]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING1]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]
// CHECK-SAME:        offsets = [0, 0], sizes = [%[[DEST_D0]], %[[DEST_D1]]], strides = [1, 1]
// CHECK-LABEL:   util.func public @update_encoding(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
util.func public @update_encoding(%resource: !stream.resource<*>, %total_size: index, %d0: index, %d1: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]][%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<?x?xf32, #encoding0>{%d0, %d1} in !stream.resource<*>{%total_size}
    -> tensor<?x?xf32, #encoding1>{%d0, %d1} in !stream.resource<*>{%total_size}
  util.return %0 : !stream.resource<*>
}

// -----

// This tests that only a single executable is created and it is reused by both
// dispatch ops.

// CHECK-DAG:  #[[ENCODING:.+]] = #iree_encoding.testing<>
#encoding = #iree_encoding.testing<>
// CHECK:      stream.executable private @[[$EX:.+]] {
// CHECK:         stream.executable.export public @[[$ENTRY:.+]] workgroups()
// CHECK-NOT:  stream.executable
// CHECK-LABEL:   util.func public @multi_identical_encode_ops(
// CHECK-SAME:      %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
util.func public @multi_identical_encode_ops(%resource: !stream.resource<*>, %total_size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<4x5xf32> in !stream.resource<*>{%total_size}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%total_size}
  // CHECK:      stream.async.dispatch on(#{{.+}}) @[[$EX]]::@[[$ENTRY]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %1 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %resource : tensor<4x5xf32> in !stream.resource<*>{%total_size}
    -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%total_size}
  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}
