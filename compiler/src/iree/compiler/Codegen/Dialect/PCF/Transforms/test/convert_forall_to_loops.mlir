// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-forall-to-loops)" --split-input-file | FileCheck %s

func.func @convert_forall(%arg0: tensor<?x?xi32>, %init: tensor<?x?xi32>, %d0: index) -> tensor<?x?xi32> {
  %0 = scf.forall (%id0, %id1) in (%d0, 32) shared_outs(%iter = %init) -> (tensor<?x?xi32>) {
    %slice = tensor.extract_slice %arg0[%id0, %id1] [2, 4] [1, 1] : tensor<?x?xi32> to tensor<2x4xi32>
    %dest_slice = tensor.extract_slice %iter[%id0, %id1] [2, 4] [1, 1] : tensor<?x?xi32> to tensor<2x4xi32>
    %copy = linalg.copy ins(%slice : tensor<2x4xi32>) outs(%dest_slice : tensor<2x4xi32>) -> tensor<2x4xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %copy into %iter[%id0, %id1] [2, 4] [1, 1] : tensor<2x4xi32> into tensor<?x?xi32>
    }
  }
  return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: @convert_forall(
//  CHECK-SAME:   %[[IN:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<?x?xi32>
//  CHECK-SAME:   %[[D0:[A-Za-z0-9_]+]]: index

//       CHECK:   %[[LOOP:.+]] = pcf.loop scope(#pcf.sequential) count(%c32, %[[D0]])
//       CHECK:     execute(%[[REF:.+]] = %[[INIT]])[%[[ID1:.+]]: index, %[[ID0:.+]]: index]
//       CHECK:          : (!pcf.sref<?x?xi32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<?x?xi32>) {
//       CHECK:       %[[DEST_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//       CHECK:       %[[COPY:.+]] = linalg.copy ins({{.*}}) outs(%[[DEST_SLICE]]
//       CHECK:       pcf.write_slice %[[COPY]] into %[[REF]][%[[ID0]], %[[ID1]]] [2, 4] [1, 1]
//       CHECK:       pcf.return
//       CHECK:   return %[[LOOP]]

// -----

func.func @convert_forall_multiple_results(%init: tensor<?xi32>, %init2: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %0:2 = scf.forall (%id) in (5) shared_outs(%iter = %init, %iter2 = %init2) -> (tensor<?xi32>, tensor<?xi32>) {
    %slice = tensor.extract_slice %iter[%id] [1] [1] : tensor<?xi32> to tensor<1xi32>
    %slice2 = tensor.extract_slice %iter2[%id] [1] [1] : tensor<?xi32> to tensor<1xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %iter2[%id] [1] [1] : tensor<1xi32> into tensor<?xi32>
      tensor.parallel_insert_slice %slice2 into %iter[%id] [1] [1] : tensor<1xi32> into tensor<?xi32>
    }
  }
  return %0#0, %0#1 : tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: @convert_forall_multiple_results(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<?xi32>
//  CHECK-SAME:   %[[INIT2:[A-Za-z0-9_]+]]: tensor<?xi32>

//       CHECK:   %[[LOOP:.+]]:2 = pcf.loop scope(#pcf.sequential) count(%c5)
//       CHECK:     execute(%[[REF:.+]] = %[[INIT]], %[[REF2:.+]] = %[[INIT2]])[%[[ID:.+]]: index]
//       CHECK:          : (!pcf.sref<?xi32, sync(#pcf.sequential)>, !pcf.sref<?xi32, sync(#pcf.sequential)>)
//       CHECK:         -> (tensor<?xi32>, tensor<?xi32>) {
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INIT]]
//       CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %[[INIT2]]
//       CHECK:       pcf.write_slice %[[SLICE]] into %[[REF2]][%[[ID]]] [1] [1]
//       CHECK:       pcf.write_slice %[[SLICE2]] into %[[REF]][%[[ID]]] [1] [1]
//       CHECK:       pcf.return
//       CHECK:   return %[[LOOP]]#0, %[[LOOP]]#1
