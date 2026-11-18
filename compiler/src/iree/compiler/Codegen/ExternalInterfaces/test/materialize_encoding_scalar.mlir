// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" %s | FileCheck %s

// Check for layouts with scalar operands.
// Unsigned CPU contractions can retain identity layouts. Their elementwise
// consumers may also have scalar inputs, which have no tensor layout.
#id = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<(d0, d1) -> ()>
#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [], innerTileSizes = [], outerDimsPerm = []}}>
#encoding = #iree_encoding.layout<[#layout]>
#target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, target_triple = "x86_64-xyz-xyz"}>
func.func @scalar_input(%arg: tensor<4x3xi32, #encoding>, %scale: f32) -> tensor<4x3xf32, #encoding> attributes {hal.executable.target = #target} {
  %empty = tensor.empty() : tensor<4x3xf32, #encoding>
  %result = linalg.generic {indexing_maps = [#id, #scalar, #id], iterator_types = ["parallel", "parallel"]} ins(%arg, %scale : tensor<4x3xi32, #encoding>, f32) outs(%empty : tensor<4x3xf32, #encoding>) {
  ^bb0(%a: i32, %s: f32, %out: f32):
    %f = arith.sitofp %a : i32 to f32
    %r = arith.mulf %f, %s : f32
    linalg.yield %r : f32
  } -> tensor<4x3xf32, #encoding>
  return %result : tensor<4x3xf32, #encoding>
}

// CHECK-LABEL: func.func @scalar_input(
// CHECK-SAME: %[[ARG:.+]]: tensor<4x3xi32>, %[[SCALE:.+]]: f32) -> tensor<4x3xf32>
// CHECK: %[[RESULT:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG]], %[[SCALE]] : tensor<4x3xi32>, f32)
// CHECK: ^bb0(%[[A:.+]]: i32, %[[S:.+]]: f32, %{{.+}}: f32):
// CHECK: %[[FLOAT:.+]] = arith.sitofp %[[A]] : i32 to f32
// CHECK: %[[SCALED:.+]] = arith.mulf %[[FLOAT]], %[[S]] : f32
// CHECK: linalg.yield %[[SCALED]] : f32
// CHECK: return %[[RESULT]] : tensor<4x3xf32>
