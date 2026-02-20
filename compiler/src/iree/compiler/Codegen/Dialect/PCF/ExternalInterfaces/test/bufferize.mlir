// RUN: iree-opt %s --one-shot-bufferize --split-input-file | FileCheck %s

util.func private @bufferize_generic(%d0: index, %d1: index, %d2: index, %d3: index) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %1 = bufferization.alloc_tensor(%d3) {memory_space = "foo"} : tensor<?xi32>
  %2:4 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, tensor<?xi32>{%d1}, tensor<?xi32>{%d2}, tensor<?xi32>) {
    util.optimization_barrier %id, %n, %ref, %ref_1, %ref_2, %ref_3 : index, index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @bufferize_generic(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D3:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//   CHECK-DAG:   %[[ALLOC1:.+]] = memref.alloc(%[[D3]]) {alignment = 64 : i64} : memref<?xi32, "foo">
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ALLOC]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ALLOC1]]
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @replay_bufferize_generic(%0: memref<?xi32>, %1: memref<?xi32>, %d0: index, %d1: index) {
  %2:4 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (memref<?xi32>, memref<?xi32>{%d0}, memref<?xi32>{%d1}, memref<?xi32>) {
    util.optimization_barrier %id, %n, %ref, %ref_1, %ref_2, %ref_3 : index, index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>
    pcf.return
  }
  util.optimization_barrier %2#0, %2#1, %2#2, %2#3 : memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>
  util.return
}

// Verify that replaying bufferization works.
// CHECK-LABEL: @replay_bufferize_generic(
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//       CHECK:            -> (memref<?xi32>, memref<?xi32>{%{{.*}}}, memref<?xi32>{%{{.*}}}, memref<?xi32>) {

// -----

util.func private @bufferize_generic_mixed(%d0: index, %d1: index, %d2: index, %1: memref<?xi32, "foo">) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %2:4 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index, %n: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, memref<?xi32>{%d1}, tensor<?xi32>{%d2}, memref<?xi32, "foo">) {
    util.optimization_barrier %id, %n, %ref, %ref_1, %ref_2, %ref_3 : index, index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @bufferize_generic_mixed(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[INIT1:[A-Za-z0-9]+]]: memref<?xi32, "foo">

//       CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//       CHECK:   pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ALLOC]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[INIT1]]
//  CHECK-SAME:             [%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @bufferize_loop(%d0: index, %d1: index, %d2: index, %d3: index, %n: index) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %1 = bufferization.alloc_tensor(%d3) {memory_space = "foo"} : tensor<?xi32>
  %2:4 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, tensor<?xi32>{%d1}, tensor<?xi32>{%d2}, tensor<?xi32>) {
    util.optimization_barrier %id, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @bufferize_loop(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D3:[A-Za-z0-9]+]]: index

//   CHECK-DAG:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//   CHECK-DAG:   %[[ALLOC1:.+]] = memref.alloc(%[[D3]]) {alignment = 64 : i64} : memref<?xi32, "foo">
//       CHECK:   pcf.loop scope(#pcf.test_scope) count
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ALLOC]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[ALLOC1]]
//  CHECK-SAME:             [%{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @bufferize_loop_mixed(%d0: index, %d1: index, %d2: index, %1: memref<?xi32, "foo">, %n: index) {
  %0 = bufferization.alloc_tensor(%d0) : tensor<?xi32>
  %2:4 = pcf.loop sync true scope(#pcf.test_scope) count(%n)
    execute(%ref = %0, %ref_1, %ref_2, %ref_3 = %1)[%id: index]
         : (!pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<?xi32>, memref<?xi32>{%d1}, tensor<?xi32>{%d2}, memref<?xi32, "foo">) {
    util.optimization_barrier %id, %ref, %ref_1, %ref_2, %ref_3 : index, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @bufferize_loop_mixed(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D2:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[INIT1:[A-Za-z0-9]+]]: memref<?xi32, "foo">

//       CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xi32>
//       CHECK:   pcf.loop sync true scope(#pcf.test_scope) count
//  CHECK-NEXT:     execute(%[[REF:[A-Za-z0-9_]+]] = %[[ALLOC]],
//  CHECK-SAME:             %[[REF1:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF2:[A-Za-z0-9_]+]],
//  CHECK-SAME:             %[[REF3:[A-Za-z0-9_]+]] = %[[INIT1]]
//  CHECK-SAME:             [%{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?xi32>, memref<?xi32>{%[[D1]]}, memref<?xi32>{%[[D2]]}, memref<?xi32, "foo">) {
//       CHECK:       pcf.return
//  CHECK-NEXT:     }

// -----

util.func private @write_tensor(%dst: !pcf.sref<?xi32, #pcf.test_scope>) {
  %src = bufferization.alloc_tensor() : tensor<2xi32>
  pcf.write_slice %src into %dst[1] [2] [1] : tensor<2xi32> into !pcf.sref<?xi32, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @write_tensor
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: !pcf.sref<?xi32, #pcf.test_scope>
//       CHECK:   %[[SRC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<2xi32>
//  CHECK-NEXT:   pcf.write_slice %[[SRC]] into %[[DST]][1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.test_scope>

// -----

util.func private @replay_write_tensor_bufferize(%src: memref<2xi32>, %dst: !pcf.sref<?xi32, #pcf.test_scope>) {
  pcf.write_slice %src into %dst[1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.test_scope>
  util.return
}

// CHECK-LABEL: @replay_write_tensor_bufferize
//  CHECK-NEXT:   pcf.write_slice %{{.*}} into %{{.*}}[1] [2] [1] : memref<2xi32> into !pcf.sref<?xi32, #pcf.test_scope>

// -----

util.func private @read_tensor(%src: !pcf.sref<?x?xi32, #pcf.test_scope>, %s0: index, %s1: index) -> tensor<?x?xi32> {
  %result = pcf.read_slice %src[0, 1] [%s0, %s1] [1, 1] : !pcf.sref<?x?xi32, #pcf.test_scope> to tensor<?x?xi32>
  util.return %result : tensor<?x?xi32>
}

// CHECK-LABEL: @read_tensor
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: !pcf.sref<?x?xi32, #pcf.test_scope>
//  CHECK-SAME:   %[[S0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[S1:[A-Za-z0-9]+]]: index
//  CHECK-NEXT:   %[[MEMREF:.+]] = pcf.get_memref %[[SRC]][0, 1] [%[[S0]], %[[S1]]] [1, 1] : !pcf.sref<?x?xi32, #pcf.test_scope> to memref<?x?xi32, strided<[?, ?], offset: ?>>
//  CHECK-NEXT:   %[[RESULT:.+]] = bufferization.to_tensor %[[MEMREF]] : memref<?x?xi32, strided<[?, ?], offset: ?>> to tensor<?x?xi32>
//  CHECK-NEXT:   util.return %[[RESULT]] : tensor<?x?xi32>

// -----

// Verify that read_slice uses the slice result shape (4x8), not the source
// sref shape (16x16), when building the memref type.
util.func private @read_slice_shape(%src: !pcf.sref<16x16xi32, #pcf.test_scope>) -> tensor<4x8xi32> {
  %result = pcf.read_slice %src[0, 0] [4, 8] [1, 1] : !pcf.sref<16x16xi32, #pcf.test_scope> to tensor<4x8xi32>
  util.return %result : tensor<4x8xi32>
}

// CHECK-LABEL: @read_slice_shape
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: !pcf.sref<16x16xi32, #pcf.test_scope>
//  CHECK-NEXT:   %[[MEMREF:.+]] = pcf.get_memref %[[SRC]][0, 0] [4, 8] [1, 1] : !pcf.sref<16x16xi32, #pcf.test_scope> to memref<4x8xi32, strided<[?, ?], offset: ?>>
//  CHECK-NEXT:   %[[RESULT:.+]] = bufferization.to_tensor %[[MEMREF]] : memref<4x8xi32, strided<[?, ?], offset: ?>> to tensor<4x8xi32>
//  CHECK-NEXT:   util.return %[[RESULT]] : tensor<4x8xi32>

// -----

// Verify that bufferizing a generic with an initializer preserves the
// num_leading_args property. Without this, the initializer's yielded values
// would be lost when the op is rebuilt during bufferization.
util.func private @bufferize_generic_with_initializer(%d0: index, %d1: index) {
  %0 = bufferization.alloc_tensor(%d0, %d1) : tensor<?x?xf32>
  %1 = pcf.generic scope(#pcf.test_scope) initialize {
      %c42 = arith.constant 42 : index
      pcf.yield %c42 : index
    } -> (%leading: index)
    execute(%ref = %0)[%id: index, %n: index]
         : (!pcf.sref<?x?xf32, #pcf.test_scope>)
        -> (tensor<?x?xf32>) {
    util.optimization_barrier %leading : index
    pcf.return
  }
  util.return
}

// CHECK-LABEL: @bufferize_generic_with_initializer(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[D1:[A-Za-z0-9]+]]: index
//       CHECK:   %[[ALLOC:.+]] = memref.alloc(%[[D0]], %[[D1]]) {alignment = 64 : i64} : memref<?x?xf32>
//       CHECK:   pcf.generic scope(#pcf.test_scope) initialize {
//  CHECK-NEXT:       %[[C42:.+]] = arith.constant 42
//  CHECK-NEXT:       pcf.yield %[[C42]]
//  CHECK-NEXT:     } -> (%[[LEADING:.+]]: index)
//  CHECK-NEXT:     execute(%{{.*}} = %[[ALLOC]])[%{{.*}}: index, %{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<?x?xf32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<?x?xf32>) {
//       CHECK:       util.optimization_barrier %[[LEADING]]
//       CHECK:       pcf.return
//  CHECK-NEXT:     }

// -----

// Verify that tied results use the init buffer directly (not the new op's
// result) while untied results use the new op's result. This ensures
// bufferization knows tied results alias their inits.
util.func private @bufferize_loop_tied_result_users(%d0: index, %n: index) -> (tensor<4xi32>, tensor<?xi32>) {
  %init = bufferization.alloc_tensor() : tensor<4xi32>
  %0:2 = pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref = %init, %ref_1)[%id: index]
         : (!pcf.sref<4xi32, #pcf.test_scope>, !pcf.sref<?xi32, #pcf.test_scope>)
        -> (tensor<4xi32>, tensor<?xi32>{%d0}) {
    pcf.return
  }
  util.return %0#0, %0#1 : tensor<4xi32>, tensor<?xi32>
}

// CHECK-LABEL: @bufferize_loop_tied_result_users(
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[N:[A-Za-z0-9]+]]: index
//       CHECK:   %[[INIT:.+]] = memref.alloc() {alignment = 64 : i64} : memref<4xi32>
//       CHECK:   %[[LOOP:.+]]:2 = pcf.loop scope(#pcf.test_scope) count(%[[N]])
//  CHECK-NEXT:     execute(%{{.*}} = %[[INIT]], %{{.*}})[%{{.*}}: index]
//  CHECK-NEXT:          : (!pcf.sref<4xi32, #pcf.test_scope>,
//  CHECK-SAME:             !pcf.sref<?xi32, #pcf.test_scope>)
//  CHECK-NEXT:         -> (memref<4xi32>, memref<?xi32>{%[[D0]]}) {
//       CHECK:       pcf.return
//  CHECK-NEXT:     }
// Tied result: replaced by init buffer directly, not the loop result.
//   CHECK-DAG:   bufferization.to_tensor %[[INIT]]
// Untied result: replaced by the loop op's result.
//   CHECK-DAG:   bufferization.to_tensor %[[LOOP]]#1
