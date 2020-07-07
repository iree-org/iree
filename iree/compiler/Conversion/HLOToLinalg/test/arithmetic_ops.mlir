// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors %s | IreeFileCheck %s

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @addf
func @addf(%operand: tensor<2x2xf32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.add"(%operand, %operand)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: f32, %[[OPERAND_IN2:.+]]: f32):
// CHECK-NEXT:   %[[RESULT:.+]] = addf %[[OPERAND_IN1]], %[[OPERAND_IN2]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }: tensor<{{.+}}xf32>, tensor<{{.+}}xf32> -> tensor<{{.+}}xf32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @addi
func @addi(%operand: tensor<2x2xi32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.add"(%operand, %operand)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: i32, %[[OPERAND_IN2:.+]]: i32):
// CHECK-NEXT:   %[[RESULT:.+]] = addi %[[OPERAND_IN1]], %[[OPERAND_IN2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }: tensor<{{.+}}xi32>, tensor<{{.+}}xi32> -> tensor<{{.+}}xi32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @subf
func @subf(%operand: tensor<2x2xf32>)
attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.subtract"(%operand, %operand)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: f32, %[[OPERAND_IN2:.+]]: f32):
// CHECK-NEXT:   %[[RESULT:.+]] = subf %[[OPERAND_IN1]], %[[OPERAND_IN2]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }: tensor<{{.+}}xf32>, tensor<{{.+}}xf32> -> tensor<{{.+}}xf32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @subi
func @subi(%operand: tensor<2x2xi32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.subtract"(%operand, %operand)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: i32, %[[OPERAND_IN2:.+]]: i32):
// CHECK-NEXT:   %[[RESULT:.+]] = subi %[[OPERAND_IN1]], %[[OPERAND_IN2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }: tensor<{{.+}}xi32>, tensor<{{.+}}xi32> -> tensor<{{.+}}xi32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @mulf
func @mulf(%operand: tensor<2x2xf32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.multiply"(%operand, %operand)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: f32, %[[OPERAND_IN2:.+]]: f32):
// CHECK-NEXT:   %[[RESULT:.+]] = mulf %[[OPERAND_IN1]], %[[OPERAND_IN2]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }: tensor<{{.+}}xf32>, tensor<{{.+}}xf32> -> tensor<{{.+}}xf32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @muli
func @muli(%operand: tensor<2x2xi32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.multiply"(%operand, %operand)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: i32, %[[OPERAND_IN2:.+]]: i32):
// CHECK-NEXT:   %[[RESULT:.+]] = muli %[[OPERAND_IN1]], %[[OPERAND_IN2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }: tensor<{{.+}}xi32>, tensor<{{.+}}xi32> -> tensor<{{.+}}xi32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @divf
func @divf(%operand: tensor<2x2xf32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.divide"(%operand, %operand)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: f32, %[[OPERAND_IN2:.+]]: f32):
// CHECK-NEXT:   %[[RESULT:.+]] = divf %[[OPERAND_IN1]], %[[OPERAND_IN2]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }: tensor<{{.+}}xf32>, tensor<{{.+}}xf32> -> tensor<{{.+}}xf32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @divi
func @divi(%operand: tensor<2x2xi32>) attributes {iree.dispatch_fn_name = ""} {
  %result = "mhlo.divide"(%operand, %operand)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return
}
// CHECK: linalg.generic {
// CHECK-SAME: args_in = 2
// CHECK-SAME: args_out = 1
// CHECK-SAME: indexing_maps
// CHECK-SAME: #[[MAP0]], #[[MAP0]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: %{{.+}}, %{{.+}} {
// CHECK-NEXT: ^{{.+}}(%[[OPERAND_IN1:.+]]: i32, %[[OPERAND_IN2:.+]]: i32):
// CHECK-NEXT:   %[[RESULT:.+]] = divi_signed %[[OPERAND_IN1]], %[[OPERAND_IN2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }: tensor<{{.+}}xi32>, tensor<{{.+}}xi32> -> tensor<{{.+}}xi32>
