// RUN: iree-opt -split-input-file -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false},canonicalize' -iree-hal-target-backends=vmvx %s | IreeFileCheck %s

#map = affine_map<(d0) -> (d0)>
flow.executable @add_dispatch_0 {
  flow.dispatch.entry @entry attributes {
    workgroup_rank = 3 : index
  }
  module  {
    func @entry(%arg0: !flow.dispatch.tensor<readonly:16xf32>, %arg1: !flow.dispatch.tensor<readonly:16xf32>, %arg2: !flow.dispatch.tensor<writeonly:16xf32>) {
      %0 = linalg.init_tensor [16] : tensor<16xf32>
      %1 = flow.dispatch.tensor.load %arg0, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %2 = flow.dispatch.tensor.load %arg1, offsets=[], sizes=[], strides=[] : !flow.dispatch.tensor<readonly:16xf32> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %2 : tensor<16xf32>, tensor<16xf32>) outs(%0 : tensor<16xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
        %4 = addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> tensor<16xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets=[], sizes=[], strides=[] : tensor<16xf32> -> !flow.dispatch.tensor<writeonly:16xf32>
      return
    }
  }
}

// CHECK-LABEL: hal.executable @add_dispatch_0
//  CHECK-NEXT:   hal.interface @io {
//  CHECK-NEXT:    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:    hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
//  CHECK-NEXT:    hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT:   }
//  CHECK-NEXT:   hal.executable.variant @vmvx, target="vmvx" {
//  CHECK-NEXT:     hal.executable.entry_point @entry attributes {
//  CHECK-SAME:       interface = @io,
//  CHECK-SAME:       ordinal = 0 : index
//  CHECK-SAME:     }
//       CHECK:     module {
//  CHECK-NEXT:       vm.module @module {
//  CHECK-NEXT:         vm.func @entry(
//  CHECK-SAME:             %[[SCRATCHPAD:.+]]: !vm.buffer, %[[CONSTANTS:.+]]: !vm.buffer,
//  CHECK-SAME:             %[[BINDINGS:.+]]: !vm.list<!vm.buffer>
//   CHECK-DAG:           %c16 = vm.const.i32 16 : i32
//   CHECK-DAG:           %zero = vm.const.i32.zero : i32
//   CHECK-DAG:           %c1 = vm.const.i32 1 : i32
//   CHECK-DAG:           %c2 = vm.const.i32 2 : i32
//   CHECK-DAG:           %c4 = vm.const.i32 4 : i32
//  CHECK-NEXT:           %[[LHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %zero : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RHS_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %c1 : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           %[[RET_BUF:.+]] = vm.list.get.ref %[[BINDINGS]], %c2 : (!vm.list<!vm.buffer>, i32) -> !vm.buffer
//  CHECK-NEXT:           vm.br ^bb1(%zero : i32)
//  CHECK-NEXT:         ^bb1(%[[IDX:.+]]: i32):  // 2 preds: ^bb0, ^bb2
//  CHECK-NEXT:           %slt = vm.cmp.lt.i32.s %[[IDX]], %c16 : i32
//  CHECK-NEXT:           vm.cond_br %slt, ^bb2, ^bb3
//  CHECK-NEXT:         ^bb2:  // pred: ^bb1
//  CHECK-NEXT:           %[[BYTE_OFFSET:.+]] = vm.mul.i32 %[[IDX]], %c4 : i32
//  CHECK-NEXT:           %[[LHS:.+]] = vm.buffer.load.f32 %[[LHS_BUF]][%[[BYTE_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RHS:.+]] = vm.buffer.load.f32 %[[RHS_BUF]][%[[BYTE_OFFSET]]] : !vm.buffer -> f32
//  CHECK-NEXT:           %[[RET:.+]] = vm.add.f32 %[[LHS]], %[[RHS]] : f32
//  CHECK-NEXT:           vm.buffer.store.f32 %[[RET]], %[[RET_BUF]][%[[BYTE_OFFSET]]] : f32 -> !vm.buffer
//  CHECK-NEXT:           %[[NEXT_IDX:.+]] = vm.add.i32 %[[IDX]], %c1 : i32
//  CHECK-NEXT:           vm.br ^bb1(%[[NEXT_IDX]] : i32)
//  CHECK-NEXT:         ^bb3:  // pred: ^bb1
//  CHECK-NEXT:           vm.return
//  CHECK-NEXT:         }
//  CHECK-NEXT:         vm.export @entry
