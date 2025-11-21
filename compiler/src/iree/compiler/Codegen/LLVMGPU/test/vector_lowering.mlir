// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-lowering,canonicalize,cse))" --split-input-file %s | FileCheck %s


func.func @multi_reduction_f16_to_single_chain_fma(%a: vector<2x1x8xf16>, %b: vector<2x1x8xf16>) -> vector<2x1xf16> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x1xf16>
  %0 = arith.mulf %a, %b : vector<2x1x8xf16>
  %1 = vector.multi_reduction <add>, %0, %cst [2] : vector<2x1x8xf16> to vector<2x1xf16>
  return %1 : vector<2x1xf16>
}

// CHECK-LABEL: func.func @multi_reduction_f16_to_single_chain_fma
// CHECK:  %[[C0:.+]]  = arith.constant dense<0.000000e+00> : vector<2xf16>
// CHECK:  %[[FMA0:.*]] = math.fma %{{.*}}, %{{.*}}, %[[C0]] : vector<2xf16>
// CHECK:  %[[FMA1:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA0]] : vector<2xf16>
// CHECK:  %[[FMA2:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA1]] : vector<2xf16>
// CHECK:  %[[FMA3:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA2]] : vector<2xf16>
// CHECK:  %[[FMA4:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA3]] : vector<2xf16>
// CHECK:  %[[FMA5:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA4]] : vector<2xf16>
// CHECK:  %[[FMA6:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA5]] : vector<2xf16>
// CHECK:  %[[FMA7:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA6]] : vector<2xf16>
// CHECK:  return %{{.*}} : vector<2x1xf16>

// -----

func.func @multi_reduction_f32_to_double_chain_fma(%a: vector<4x1x8xf32>, %b: vector<4x1x8xf32>) -> vector<4x1xf32> {
 %cst = arith.constant dense<0.000000e+00> : vector<4x1xf32>
 %0 = arith.mulf %a, %b : vector<4x1x8xf32>
 %1 = vector.multi_reduction <add>, %0, %cst [2] : vector<4x1x8xf32> to vector<4x1xf32>
 return %1 : vector<4x1xf32>
}

// CHECK-LABEL: func.func @multi_reduction_f32_to_double_chain_fma
// CHECK:  %[[C0:.+]]  = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK:  %[[FMA0:.*]] = math.fma %{{.*}}, %{{.*}}, %[[C0]] : vector<4xf32>
// CHECK:  %[[FMA1:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA0]] : vector<4xf32>
// CHECK:  %[[FMA2:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA1]] : vector<4xf32>
// CHECK:  %[[FMA3:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA2]] : vector<4xf32>
// CHECK:  %[[FMA4:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA3]] : vector<4xf32>
// CHECK:  %[[FMA5:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA4]] : vector<4xf32>
// CHECK:  %[[FMA6:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA5]] : vector<4xf32>
// CHECK:  %[[FMA7:.*]] = math.fma %{{.*}}, %{{.*}}, %[[FMA6]] : vector<4xf32>
// CHECK:  return %{{.*}} : vector<4x1xf32>

// -----

#lhs = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#res = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @general_contract_add_to_chain_fma(
    %A: vector<3x4x2xf32>,
    %B: vector<4x3x2xf32>) -> vector<3x2xf32> {
  %c0 = arith.constant dense<0.000000e+00> : vector<3x2xf32>
  %out = vector.contract
           { indexing_maps = [#lhs, #rhs, #res],
             iterator_types = ["parallel","parallel","reduction"],
             kind = #vector.kind<add> }
           %A, %B, %c0
         : vector<3x4x2xf32>, vector<4x3x2xf32> into vector<3x2xf32>
  return %out : vector<3x2xf32>
}

// CHECK-LABEL: func.func @general_contract_add_to_chain_fma
// CHECK:  %[[C0:.+]] = arith.constant dense<0.000000e+00> : vector<6xf32>
// CHECK:  %[[FMA0:.*]] = math.fma {{.*}}, {{.*}}, %[[C0]] : vector<6xf32>
// CHECK:  %[[FMA1:.*]] = math.fma {{.*}}, {{.*}}, %[[FMA0]] : vector<6xf32>
// CHECK:  %[[FMA2:.*]] = math.fma {{.*}}, {{.*}}, %[[FMA1]] : vector<6xf32>
// CHECK:  return %{{.*}} : vector<3x2xf32>

// -----

#lhs = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#res = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @int_contract_add_no_chain_fma(
    %A: vector<3x4x2xi32>,
    %B: vector<4x3x2xi32>) -> vector<3x2xi32> {
  %c0 = arith.constant dense<0> : vector<3x2xi32>
  %out = vector.contract
           { indexing_maps = [#lhs, #rhs, #res],
             iterator_types = ["parallel","parallel","reduction"],
             kind = #vector.kind<add> }
           %A, %B, %c0
         : vector<3x4x2xi32>, vector<4x3x2xi32> into vector<3x2xi32>
  return %out : vector<3x2xi32>
}

// CHECK-LABEL: func.func @int_contract_add_no_chain_fma
// CHECK-NOT:  math.fma
// CHECK:  return %{{.*}} : vector<3x2xi32>
