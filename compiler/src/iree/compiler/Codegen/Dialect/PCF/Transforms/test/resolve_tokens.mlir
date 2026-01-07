// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-resolve-tokens)" --split-input-file | FileCheck %s

func.func @convert_generic(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.test_scope)>)
        -> (memref<?x?xi32>) {
    util.optimization_barrier %ref : !pcf.sref<?x?xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  return
}

// CHECK-LABEL: @convert_generic
//       CHECK:   pcf.generic sync true
//  CHECK-NEXT:     execute(%[[REF:.+]] = %{{.*}})
//  CHECK-NEXT:          : (!pcf.sref<?x?xi32, #pcf.test_scope>)
//       CHECK:       util.optimization_barrier %[[REF]]

// -----

func.func @convert_generic_block_args(%arg0: memref<?x?xi32>) {
  pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id: index, %n: index]
         : (!pcf.sref<?x?xi32, sync(#pcf.test_scope)>)
        -> (memref<?x?xi32>) {
    cf.br ^bb1(%ref : !pcf.sref<?x?xi32, sync(#pcf.test_scope)>)
   ^bb1(%ref_0: !pcf.sref<?x?xi32, sync(#pcf.test_scope)>):
    util.optimization_barrier %ref_0 : !pcf.sref<?x?xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  return
}

// CHECK-LABEL: @convert_generic_block_args
//       CHECK:   pcf.generic sync true
//  CHECK-NEXT:     execute(%[[REF:.+]] = %{{.*}})
//  CHECK-NEXT:          : (!pcf.sref<?x?xi32, #pcf.test_scope>)
//       CHECK:     cf.br ^bb1(%[[REF]]
//       CHECK:    ^bb1(%[[BRANCH:.+]]: !pcf.sref<?x?xi32, #pcf.test_scope>):
//       CHECK:       util.optimization_barrier %[[BRANCH]]

// -----

func.func @do_not_sync_no_result_generic() {
  pcf.generic scope(#pcf.test_scope)
    execute[%id: index, %count: index] {
    pcf.return
  }
  return
}

// CHECK-LABEL: @do_not_sync_no_result_generic
//       CHECK:   pcf.generic scope

// -----

func.func @convert_loop(%arg0: memref<?x?xi32>, %n: index) {
  pcf.loop scope(#pcf.test_scope) count(%n)
    execute(%ref = %arg0)[%id: index]
            : (!pcf.sref<?x?xi32, sync(#pcf.test_scope)>)
           -> (memref<?x?xi32>) {
    util.optimization_barrier %ref : !pcf.sref<?x?xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  return
}

// CHECK-LABEL: @convert_loop
//       CHECK:   pcf.loop sync true
//  CHECK-NEXT:     execute(%[[REF:.+]] = %{{.*}})
//  CHECK-NEXT:          : (!pcf.sref<?x?xi32, #pcf.test_scope>)
//       CHECK:       util.optimization_barrier %[[REF]]

// -----

func.func @do_not_sync_no_result_loop(%n: index) {
  pcf.loop scope(#pcf.test_scope) count(%n)
    execute[%id: index] {
    pcf.return
  }
  return
}

// CHECK-LABEL: @do_not_sync_no_result_loop
//       CHECK:   pcf.loop scope

// -----

func.func @convert_write_slice(%arg0: memref<3x4xi32>, %ref: !pcf.sref<?x?xi32, sync(#pcf.test_scope)>) {
  pcf.write_slice %arg0 into %ref[1, 2] [3, 4] [1, 1] : memref<3x4xi32> into !pcf.sref<?x?xi32, sync(#pcf.test_scope)>
  return
}

// CHECK-LABEL: @convert_write_slice
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: memref<3x4xi32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: !pcf.sref<?x?xi32, #pcf.test_scope>
//       CHECK:   pcf.write_slice %[[ARG0]] into %[[ARG1]][1, 2] [3, 4] [1, 1]
//  CHECK-SAME:    : memref<3x4xi32> into !pcf.sref<?x?xi32, #pcf.test_scope>

// -----

func.func @convert_alloc(%d0: index) -> !pcf.sref<?x5xi32, sync(#pcf.test_scope)> {
  %0 = pcf.alloc(%d0) : !pcf.sref<?x5xi32, sync(#pcf.test_scope)>
  return %0 : !pcf.sref<?x5xi32, sync(#pcf.test_scope)>
}

// CHECK-LABEL: @convert_alloc
//  CHECK-SAME:   %[[D0:[A-Za-z0-9]+]]: index
//       CHECK:   %[[ALLOC:.+]] = pcf.alloc(%[[D0]]) : !pcf.sref<?x5xi32, #pcf.test_scope>
//       CHECK:   return %[[ALLOC]] : !pcf.sref<?x5xi32, #pcf.test_scope>
