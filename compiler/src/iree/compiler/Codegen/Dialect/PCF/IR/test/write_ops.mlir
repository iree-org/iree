// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @write_tensor(%src: tensor<?x?xi32>, %dst: !pcf.sref<?x?xi32, #pcf.dummy_scope>, %s0: index, %s1: index) {
  pcf.write_slice %src into %dst[0, 1] [%s0, %s1] [1, 1] : tensor<?x?xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
  util.return
}

// CHECK-LABEL: @write_tensor
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: !pcf.sref<?x?xi32, #pcf.dummy_scope>
//  CHECK-SAME:   %[[S0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[S1:[A-Za-z0-9]+]]: index
//  CHECK-NEXT:   pcf.write_slice %[[SRC]] into %[[DST]][0, 1] [%[[S0]], %[[S1]]] [1, 1] : tensor<?x?xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>

// -----

util.func private @write_vector(%src: vector<3x4xi32>, %dst: !pcf.sref<?x?xi32, #pcf.dummy_scope>, %o0: index, %o1: index) {
  pcf.write_slice %src into %dst[%o0, %o1] [3, 4] [1, 1] : vector<3x4xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
  util.return
}

// CHECK-LABEL: @write_vector
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: vector<3x4xi32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: !pcf.sref<?x?xi32, #pcf.dummy_scope>
//  CHECK-SAME:   %[[O0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[O1:[A-Za-z0-9]+]]: index
//  CHECK-NEXT:   pcf.write_slice %[[SRC]] into %[[DST]][%[[O0]], %[[O1]]] [3, 4] [1, 1] : vector<3x4xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>

// -----

util.func private @write_memref(%src: memref<?x?xi32>, %dst: !pcf.sref<?x?xi32, #pcf.dummy_scope>, %s0: index, %s1: index) {
  pcf.write_slice %src into %dst[0, 1] [%s0, %s1] [1, 1] : memref<?x?xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
  util.return
}

// CHECK-LABEL: @write_memref
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: memref<?x?xi32>
//  CHECK-SAME:   %[[DST:[A-Za-z0-9]+]]: !pcf.sref<?x?xi32, #pcf.dummy_scope>
//  CHECK-SAME:   %[[S0:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[S1:[A-Za-z0-9]+]]: index
//  CHECK-NEXT:   pcf.write_slice %[[SRC]] into %[[DST]][0, 1] [%[[S0]], %[[S1]]] [1, 1] : memref<?x?xi32> into !pcf.sref<?x?xi32, #pcf.dummy_scope>
