// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-attention{skip-masked-tiles=true}))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-attention))" --split-input-file %s | FileCheck %s --check-prefix=DEFAULT

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @bool_mask(%q: tensor<2x3x4xf32>, %k: tensor<2x5x4xf32>, %v: tensor<2x5x6xf32>,
    %mask: tensor<2x3x5xi1>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x5x6xf32>, f32, tensor<2x3x5xi1>)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @bool_mask
//  CHECK-SAME:     %[[Q:[a-zA-Z0-9]+]]: tensor<2x3x4xf32>, %[[K:[a-zA-Z0-9]+]]: tensor<2x5x4xf32>, %[[V:[a-zA-Z0-9]+]]: tensor<2x5x6xf32>,
//  CHECK-SAME:     %[[MASK:[a-zA-Z0-9]+]]: tensor<2x3x5xi1>,
//  CHECK-SAME:     %[[ACC:[a-zA-Z0-9]+]]: tensor<2x3x6xf32>, %[[MAX:[a-zA-Z0-9]+]]: tensor<2x3xf32>, %[[SUM:[a-zA-Z0-9]+]]: tensor<2x3xf32>)
//       CHECK:   %[[ANY:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[MASK]] : tensor<2x3x5xi1>) outs(%{{.+}} : tensor<i1>)
//       CHECK:     ^bb0(%[[ENTRY:.+]]: i1, %[[SEEN:.+]]: i1):
//       CHECK:       %[[OR:.+]] = arith.ori %[[ENTRY]], %[[SEEN]] : i1
//       CHECK:       linalg.yield %[[OR]] : i1
//       CHECK:   %[[VISIBLE:.+]] = tensor.extract %[[ANY]][] : tensor<i1>
//       CHECK:   %[[IF:.+]]:3 = scf.if %[[VISIBLE]]
//       CHECK:     linalg.generic {{.*}} ins(%{{.+}}, %[[K]] :
//       CHECK:     math.exp2
//       CHECK:     linalg.generic {{.*}} ins(%{{.+}}, %[[V]] :
//       CHECK:     scf.yield
//       CHECK:   } else {
//  CHECK-NEXT:     scf.yield %[[ACC]], %[[MAX]], %[[SUM]]
//       CHECK:   return %[[IF]]#0, %[[IF]]#1, %[[IF]]#2

// DEFAULT-LABEL: func.func @bool_mask
//   DEFAULT-NOT:   scf.if

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @additive_mask(%q: tensor<2x3x4xf32>, %k: tensor<2x5x4xf32>, %v: tensor<2x5x6xf32>,
    %mask: tensor<2x3x5xf32>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x5x6xf32>, f32, tensor<2x3x5xf32>)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @additive_mask
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%{{.+}} : tensor<i1>)
//       CHECK:     ^bb0(%[[ENTRY:.+]]: f32, %[[SEEN:.+]]: i1):
//       CHECK:       %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
//       CHECK:       %[[KEPT:.+]] = arith.cmpf une, %[[ENTRY]], %[[NEGINF]] : f32
//       CHECK:       arith.ori %[[KEPT]], %[[SEEN]] : i1
//       CHECK:   scf.if

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @i8_mask(%q: tensor<2x3x4xf32>, %k: tensor<2x5x4xf32>, %v: tensor<2x5x6xf32>,
    %mask: tensor<2x3x5xi8>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x5x6xf32>, f32, tensor<2x3x5xi8>)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @i8_mask
//       CHECK:   linalg.generic
//  CHECK-SAME:       outs(%{{.+}} : tensor<i1>)
//       CHECK:     ^bb0(%[[ENTRY:.+]]: i8, %[[SEEN:.+]]: i1):
//       CHECK:       %[[BIT:.+]] = arith.trunci %[[ENTRY]] : i8 to i1
//       CHECK:       arith.ori %[[BIT]], %[[SEEN]] : i1
//       CHECK:   scf.if

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @broadcast_mask(%q: tensor<2x3x4xf32>, %k: tensor<2x5x4xf32>, %v: tensor<2x5x6xf32>,
    %mask: tensor<2x5xi1>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapB, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x5x6xf32>, f32, tensor<2x5xi1>)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @broadcast_mask
//  CHECK-SAME:     %[[MASK:[a-zA-Z0-9]+]]: tensor<2x5xi1>
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[MASK]] : tensor<2x5xi1>) outs(%{{.+}} : tensor<i1>)
//       CHECK:   scf.if

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @dynamic_keys(%q: tensor<2x3x4xf32>, %k: tensor<2x?x4xf32>, %v: tensor<2x?x6xf32>,
    %mask: tensor<2x3x?xi1>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapM, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale, %mask : tensor<2x3x4xf32>, tensor<2x?x4xf32>, tensor<2x?x6xf32>, f32, tensor<2x3x?xi1>)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @dynamic_keys
//  CHECK-SAME:     %[[MASK:[a-zA-Z0-9]+]]: tensor<2x3x?xi1>
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[MASK]] : tensor<2x3x?xi1>) outs(%{{.+}} : tensor<i1>)
//       CHECK:   scf.if

// -----

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapM = affine_map<(batch, m, k1, k2, n) -> (batch, m, k2)>
#mapB = affine_map<(batch, m, k1, k2, n) -> (batch, k2)>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
#mapR = affine_map<(batch, m, k1, k2, n) -> (batch, m)>

func.func @no_mask(%q: tensor<2x3x4xf32>, %k: tensor<2x5x4xf32>, %v: tensor<2x5x6xf32>,
    %acc: tensor<2x3x6xf32>, %max: tensor<2x3xf32>, %sum: tensor<2x3xf32>)
    -> (tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  %scale = arith.constant 0.5 : f32
  %out:3 = iree_linalg_ext.online_attention
      <indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO, #mapR, #mapR]>
      ins(%q, %k, %v, %scale : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x5x6xf32>, f32)
      outs(%acc, %max, %sum : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    ^bb0(%s: f32):
      iree_linalg_ext.yield %s : f32
  } -> tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %out#0, %out#1, %out#2 : tensor<2x3x6xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @no_mask
//   CHECK-NOT:   scf.if
//       CHECK:   return
