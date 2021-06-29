// RUN: iree-opt -allow-unregistered-dialect -split-input-file -iree-hal-materialize-interfaces -iree-hal-target-backends=vmvx %s | IreeFileCheck %s

// CHECK-LABEL: hal.executable @static_tiled_dispatch
//  CHECK-NEXT: hal.interface @[[IO:.+]] {
//  CHECK-NEXT:   hal.interface.binding @[[S0B0:.+]], set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B1:.+]], set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @static_tiled_dispatch {
  // CHECK-NEXT: hal.executable.target @vmvx, filter="vmvx" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @[[IO]],
  // CHECK-SAME:     ordinal = 0 : index
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(%arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>) {
      // CHECK-NEXT: %c0 = constant 0 : index
      // CHECK-NEXT: %[[ARG:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B0]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-NEXT: %[[RET:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B1]][%c0] : !flow.dispatch.tensor<writeonly:4x8xf32>

      // CHECK-NEXT: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG]]
      %arg_tile = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.sink"(%[[ARG_TILE]])
      %ret_tile = "test.sink"(%arg_tile) : (tensor<8x4xf32>) -> tensor<4x8xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET]]
      flow.dispatch.tensor.store %ret_tile, %ret, offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
      return
    }
  }
}
func @usage(%func_arg: tensor<8x4xf32>) -> tensor<4x8xf32> {
  %0 = flow.ex.stream.fragment(%func_arg) : (tensor<8x4xf32>) -> tensor<4x8xf32> =
      (%stream_arg: tensor<8x4xf32>) -> tensor<4x8xf32> {
    %c1 = constant 1 : index
    // CHECK: = flow.dispatch @static_tiled_dispatch::@entry
    // CHECK-SAME: hal.bindings = [
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B0]]", 0 : index>,
    // CHECK-SAME:   #hal.ex.result_buffer<"[[S0B1]]", 0 : index>]
    %1 = flow.dispatch @static_tiled_dispatch::@entry[%c1, %c1, %c1](%stream_arg) : (tensor<8x4xf32>) -> tensor<4x8xf32>
    flow.return %1 : tensor<4x8xf32>
  }
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: hal.executable @dynamic_tiled_dispatch
//  CHECK-NEXT: hal.interface @[[IO:.+]] attributes {push_constants = 4 : index} {
//  CHECK-NEXT:   hal.interface.binding @[[S0B0:.+]], set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B1:.+]], set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @dynamic_tiled_dispatch {
  // CHECK-NEXT: hal.executable.target @vmvx, filter="vmvx" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @[[IO]],
  // CHECK-SAME:     ordinal = 0 : index
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(
        // CHECK-NEXT: %c0 = constant 0 : index
        // CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B0]][%c0] : !flow.dispatch.tensor<readonly:7x?x24x?xf32>
        %arg: !flow.dispatch.tensor<readonly:7x?x24x?xf32>,
        // CHECK-DAG: %[[RET:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B1]][%c0] : !flow.dispatch.tensor<writeonly:?x?x1024xf32>
        %ret: !flow.dispatch.tensor<writeonly:?x?x1024xf32>,
        // CHECK-DAG: %[[ARG_DIM1:.+]] = hal.interface.load.constant offset = 0 : index
        %arg_dim1: index,
        // CHECK-DAG: %[[ARG_DIM3:.+]] = hal.interface.load.constant offset = 1 : index
        %arg_dim3: index,
        // CHECK-DAG: %[[RET_DIM0:.+]] = hal.interface.load.constant offset = 2 : index
        %ret_dim0: index,
        // CHECK-DAG: %[[RET_DIM1:.+]] = hal.interface.load.constant offset = 3 : index
        %ret_dim1: index
      ) {
      // CHECK-NEXT: %[[ARG_SHAPE:.+]] = shapex.make_ranked_shape %[[ARG_DIM1]], %[[ARG_DIM3]]
      %arg_shape = shapex.make_ranked_shape %arg_dim1, %arg_dim3 : (index, index) -> !shapex.ranked_shape<[7,?,24,?]>
      // CHECK-NEXT: %[[ARG_SHAPED:.+]] = flow.dispatch.tie_shape %[[ARG]], %[[ARG_SHAPE]]
      %arg_shaped = flow.dispatch.tie_shape %arg, %arg_shape : (!flow.dispatch.tensor<readonly:7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>) -> !flow.dispatch.tensor<readonly:7x?x24x?xf32>
      // CHECK-NEXT: %[[RET_SHAPE:.+]] = shapex.make_ranked_shape %[[RET_DIM0]], %[[RET_DIM1]]
      %ret_shape = shapex.make_ranked_shape %ret_dim0, %ret_dim1 : (index, index) -> !shapex.ranked_shape<[?,?,1024]>
      // CHECK-NEXT: %[[RET_SHAPED:.+]] = flow.dispatch.tie_shape %[[RET]], %[[RET_SHAPE]]
      %ret_shaped = flow.dispatch.tie_shape %ret, %ret_shape : (!flow.dispatch.tensor<writeonly:?x?x1024xf32>, !shapex.ranked_shape<[?,?,1024]>) -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>
      // CHECK-NEXT: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_SHAPED]]
      %arg_tile = flow.dispatch.tensor.load %arg_shaped, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:7x?x24x?xf32> -> tensor<7x?x24x?xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]])
      %ret_tile = "test.tile_math"(%arg_tile) : (tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_SHAPED]]
      flow.dispatch.tensor.store %ret_tile, %ret_shaped, offsets=[], sizes=[], strides=[] : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>
      return
    }
  }
}
func @usage(%func_arg: tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %d0 = constant 100 : index
  %d1 = constant 200 : index
  %0 = flow.ex.stream.fragment(%func_arg, %d0, %d1) : (tensor<7x?x24x?xf32>{%d0, %d1}, index, index) -> tensor<?x?x1024xf32>{%d1, %d0} =
      (%stream_arg: tensor<7x?x24x?xf32>, %stream_d0: index, %stream_d1: index) -> tensor<?x?x1024xf32> {
    %c1 = constant 1 : index
    // CHECK: = flow.dispatch @dynamic_tiled_dispatch::@entry
    // CHECK-SAME: hal.bindings = [
    // CHECK-SAME:   #hal.ex.push_constant<0 : index, 1 : index>,
    // CHECK-SAME:   #hal.ex.push_constant<1 : index, 2 : index>,
    // CHECK-SAME:   #hal.ex.push_constant<2 : index, 3 : index>,
    // CHECK-SAME:   #hal.ex.push_constant<3 : index, 4 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B0]]", 0 : index>,
    // CHECK-SAME:   #hal.ex.result_buffer<"[[S0B1]]", 0 : index>
    %1 = flow.dispatch @dynamic_tiled_dispatch::@entry[%c1, %c1, %c1](%stream_arg, %stream_d0, %stream_d1, %stream_d1, %stream_d0)
        : (tensor<7x?x24x?xf32>{%stream_d0, %stream_d1}, index, index, index, index) -> tensor<?x?x1024xf32>{%stream_d1, %stream_d0}
    flow.return %1 : tensor<?x?x1024xf32>
  }
  return %0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: hal.executable @workgroup_infos
//  CHECK-NEXT: hal.interface @[[IO:.+]] {
//  CHECK-NEXT:   hal.interface.binding @[[S0B0:.+]], set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B1:.+]], set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @workgroup_infos {
  // CHECK-NEXT: hal.executable.target @vmvx, filter="vmvx" {
  // CHECK-NEXT:   hal.executable.entry_point @entry attributes {
  // CHECK-SAME:     interface = @[[IO]],
  // CHECK-SAME:     ordinal = 0 : index
  // CHECK-SAME:   }
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 2 : index
  }
  // CHECK-NEXT: module  {
  module  {
    // CHECK-NEXT: func @entry() {
    func @entry(%arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>) {
      // CHECK-DAG: %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
      %id_x = flow.dispatch.workgroup.id[0] : index
      // CHECK-DAG: %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
      %id_y = flow.dispatch.workgroup.id[1] : index
      // CHECK-DAG: %[[WORKGROUP_COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
      %count_x = flow.dispatch.workgroup.count[0] : index
      // CHECK-DAG: %[[WORKGROUP_COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index
      %count_y = flow.dispatch.workgroup.count[1] : index
      // CHECK-DAG: %[[WORKGROUP_SIZE_X:.+]] = hal.interface.workgroup.size[0] : index
      %size_x = flow.dispatch.workgroup.size[0] : index
      // CHECK-DAG: %[[WORKGROUP_SIZE_Y:.+]] = hal.interface.workgroup.size[1] : index
      %size_y = flow.dispatch.workgroup.size[1] : index
      // CHECK-NEXT: "test.sink"(%[[WORKGROUP_ID_X]], %[[WORKGROUP_ID_Y]], %[[WORKGROUP_COUNT_X]], %[[WORKGROUP_COUNT_Y]], %[[WORKGROUP_SIZE_X]], %[[WORKGROUP_SIZE_Y]])
      "test.sink"(%id_x, %id_y, %count_x, %count_y, %size_x, %size_y) : (index, index, index, index, index, index) -> ()
      return
    }
  }
}

// -----

// CHECK-LABEL: hal.executable @static_tied_result
//  CHECK-NEXT: hal.interface @[[IO:.+]] {
//  CHECK-NEXT:   hal.interface.binding @[[S0B0:.+]], set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B1:.+]], set=0, binding=1, type="StorageBuffer", access="Read|Write"
//  CHECK-NEXT: }
flow.executable @static_tied_result {
  // CHECK-NEXT: hal.executable.target @vmvx, filter="vmvx" {
  // CHECK-NEXT:   hal.executable.entry_point @entry
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 2 : index
  }
  module  {
    // CHECK: func @entry() {
    func @entry(%arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<readwrite:4x8xf32>) {
      // CHECK-NEXT: %c0 = constant 0 : index
      // CHECK-NEXT: %[[ARG:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B0]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-NEXT: %[[RET:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B1]][%c0] : !flow.dispatch.tensor<readwrite:4x8xf32>
      // CHECK-NEXT: %[[ARG0_TILE:.+]] = flow.dispatch.tensor.load %[[ARG]]
      %arg0_tile = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[ARG1_TILE:.+]] = flow.dispatch.tensor.load %[[RET]]
      %arg1_tile = flow.dispatch.tensor.load %ret, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readwrite:4x8xf32> -> tensor<4x8xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.sink"(%[[ARG0_TILE]], %[[ARG1_TILE]])
      %ret_tile = "test.sink"(%arg0_tile, %arg1_tile) : (tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET]]
      flow.dispatch.tensor.store %ret_tile, %ret, offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<readwrite:4x8xf32>
      return
    }
  }
}
func @usage(%func_arg: tensor<8x4xf32>, %func_ret: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = flow.ex.stream.fragment(%func_arg, %func_ret) : (tensor<8x4xf32>, tensor<4x8xf32>) -> %func_ret =
      (%stream_arg: tensor<8x4xf32>, %stream_ret: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %c1 = constant 1 : index
    // CHECK: = flow.dispatch @static_tied_result::@entry
    // CHECK-SAME: hal.bindings = [
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B0]]", 0 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B1]]", 1 : index>
    %1 = flow.dispatch @static_tied_result::@entry[%c1, %c1, %c1](%stream_arg, %stream_ret) : (tensor<8x4xf32>, tensor<4x8xf32>) -> %stream_ret
    flow.return %1 : tensor<4x8xf32>
  }
  return %0 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: hal.executable @constant_dispatch
//  CHECK-NEXT: hal.interface @[[IO:.+]] {
//  CHECK-NEXT:   hal.interface.binding @[[S0B0:.+]], set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B1:.+]], set=0, binding=1, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B2:.+]], set=0, binding=2, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B3:.+]], set=0, binding=3, type="StorageBuffer", access="Read"
//  CHECK-NEXT:   hal.interface.binding @[[S0B4:.+]], set=0, binding=4, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT: }
flow.executable @constant_dispatch {
  // CHECK-NEXT: hal.executable.target @vmvx, filter="vmvx" {
  // CHECK-NEXT:   hal.executable.entry_point @entry
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 2 : index
  }
  module  {
    // CHECK: func @entry() {
    func @entry(
        %arg: !flow.dispatch.tensor<readonly:8x4xf32>,
        %const_span_0: !flow.dispatch.tensor<readonly:8x4xf32>,
        %const_span_1: !flow.dispatch.tensor<readonly:8x4xf32>,
        %const_span_2: !flow.dispatch.tensor<readonly:8x4xf32>,
        %const_span_3: !flow.dispatch.tensor<readonly:8x4xf32>,
        %ret: !flow.dispatch.tensor<writeonly:4x8xf32>) {
      // CHECK-DAG: %[[ARG:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B1]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-DAG: %[[CONST_0:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B0]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-DAG: %[[CONST_1:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B0]][%c128] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-DAG: %[[CONST_2:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B2]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-DAG: %[[CONST_3:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B3]][%c0] : !flow.dispatch.tensor<readonly:8x4xf32>
      // CHECK-DAG: %[[RET:.+]] = hal.interface.binding.subspan @[[IO]]::@[[S0B4]][%c0] : !flow.dispatch.tensor<writeonly:4x8xf32>

      // CHECK-NEXT: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG]]
      %arg_tile = flow.dispatch.tensor.load %arg, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[CONST_TILE_0:.+]] = flow.dispatch.tensor.load %[[CONST_0]]
      %const_tile_0 = flow.dispatch.tensor.load %const_span_0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[CONST_TILE_1:.+]] = flow.dispatch.tensor.load %[[CONST_1]]
      %const_tile_1 = flow.dispatch.tensor.load %const_span_1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[CONST_TILE_2:.+]] = flow.dispatch.tensor.load %[[CONST_2]]
      %const_tile_2 = flow.dispatch.tensor.load %const_span_2, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[CONST_TILE_3:.+]] = flow.dispatch.tensor.load %[[CONST_3]]
      %const_tile_3 = flow.dispatch.tensor.load %const_span_3, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
      // CHECK-NEXT: %[[RET_TILE:.+]] = "test.sink"
      // CHECK-SAME:   (%[[ARG_TILE]], %[[CONST_TILE_0]], %[[CONST_TILE_1]], %[[CONST_TILE_2]], %[[CONST_TILE_3]])
      %ret_tile = "test.sink"
          (%arg_tile, %const_tile_0, %const_tile_1, %const_tile_2, %const_tile_3) :
          (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<4x8xf32>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET]]
      flow.dispatch.tensor.store %ret_tile, %ret, offsets=[], sizes=[], strides=[] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
      return
    }
  }
}
hal.variable @storage0 : !hal.buffer
hal.variable @storage1 : !hal.buffer
func @usage(%func_arg: tensor<8x4xf32>) -> tensor<4x8xf32> {
  %0 = flow.ex.stream.fragment(%func_arg) : (tensor<8x4xf32>) -> tensor<4x8xf32> =
      (%stream_arg: tensor<8x4xf32>) -> tensor<4x8xf32> {
    %const_span_0a = hal.constant.subspan @storage0[#hal.byte_range<0, 128>] : tensor<8x4xf32>
    %const_span_0b = hal.constant.subspan @storage0[#hal.byte_range<128, 128>] : tensor<8x4xf32>
    %const_span_0c = hal.constant.subspan @storage0[#hal.byte_range<256, 128>] : tensor<8x4xf32>
    %const_span_1a = hal.constant.subspan @storage1[#hal.byte_range<0, 128>] : tensor<8x4xf32>
    %c1 = constant 1 : index
    // CHECK: = flow.dispatch @constant_dispatch::@entry
    // CHECK-SAME: hal.bindings = [
    // CHECK-SAME:   #hal.ex.constant_storage<"[[S0B0]]", "storage0", 0 : index, 256 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B1]]", 0 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B2]]", 3 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B3]]", 4 : index>,
    // CHECK-SAME:   #hal.ex.result_buffer<"[[S0B4]]", 0 : index>]
    %1 = flow.dispatch @constant_dispatch::@entry[%c1, %c1, %c1]
        (%stream_arg, %const_span_0a, %const_span_0b, %const_span_0c, %const_span_1a) :
        (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<4x8xf32>
    // NOTE: transposed to prevent %const_span_0c from being uniform.
    // CHECK: = flow.dispatch @constant_dispatch::@entry
    // CHECK-SAME: hal.bindings = [
    // CHECK-SAME:   #hal.ex.constant_storage<"[[S0B0]]", "storage0", 0 : index, 256 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B1]]", 0 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B2]]", 3 : index>,
    // CHECK-SAME:   #hal.ex.operand_buffer<"[[S0B3]]", 4 : index>,
    // CHECK-SAME:   #hal.ex.result_buffer<"[[S0B4]]", 0 : index>]
    %2 = flow.dispatch @constant_dispatch::@entry[%c1, %c1, %c1]
        (%stream_arg, %const_span_0a, %const_span_0b, %const_span_1a, %const_span_0c) :
        (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<4x8xf32>
    flow.return %2 : tensor<4x8xf32>
  }
  return %0 : tensor<4x8xf32>
}
