// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-vmvx-lower-linalg-microkernels, canonicalize, cse))" %s | FileCheck %s

// Verifies the indexing math generated in order to resolve subviews to 1D.
// This incidentally also verifies vmvx.copy (non-transposed) lowering.
// CHECK-LABEL: @subview_indexing_2d
//   CHECK-DAG: %[[ARG0SV:.*]] = memref.subview %arg0
//   CHECK-DAG: %[[ARG1SV:.*]] = memref.subview %arg1
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %[[ARG0SV]]
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZES1:.*]]:2, %[[STRIDES1:.*]]:2 = vmvx.get_buffer_descriptor %[[ARG1SV]]
//       CHECK: vmvx.copy in(%[[BB1]] offset %[[OFFSET1]] strides[%[[STRIDES1]]#0, %[[STRIDES1]]#1] : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] strides[%[[STRIDES0]]#0, %[[STRIDES0]]#1] : !util.buffer)
//  CHECK-SAME:   sizes(%[[SIZES0]]#0, %[[SIZES0]]#1)
func.func @subview_indexing_2d(%arg0 : memref<384x128xf32>, %arg1 : memref<128x384xf32>, %arg2 : index, %arg3 : index) {
  %6 = memref.subview %arg0[%arg2, %arg3] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
  %7 = memref.subview %arg1[%arg3, %arg2] [64, 64] [1, 1] : memref<128x384xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
  // A non-broadcasting 2d copy.
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%7 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
    outs(%6 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) {
  ^bb0(%arg4: f32, %arg5: f32):
    linalg.yield %arg4 : f32
  }
  func.return
}

// Verifies that 2d generic with swapped dims lowers to vmvx.copy with swapped
// strides.
// CHECK-LABEL: @generic_2d_transposed_to_copy
//   CHECK-DAG: %[[ARG0SV:.*]] = memref.subview %arg0
//   CHECK-DAG: %[[ARG1SV:.*]] = memref.subview %arg1
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %[[ARG0SV]]
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZES1:.*]]:2, %[[STRIDES1:.*]]:2 = vmvx.get_buffer_descriptor %[[ARG1SV]]
//       CHECK: vmvx.copy in({{.*}} offset {{.*}} strides[%[[STRIDES1]]#1, %[[STRIDES1]]#0] : !util.buffer)
//  CHECK-SAME:   out({{.*}} offset {{.*}} strides[%[[STRIDES0]]#0, %[[STRIDES0]]#1] : !util.buffer) sizes({{.*}})
func.func @generic_2d_transposed_to_copy(%arg0 : memref<384x128xf32>, %arg1 : memref<128x384xf32>, %arg2 : index, %arg3 : index) {
  %6 = memref.subview %arg0[%arg2, %arg3] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>
  %7 = memref.subview %arg1[%arg3, %arg2] [64, 64] [1, 1] : memref<128x384xf32> to memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>
  // A transposed 2d copy.
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%7 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>>)
    outs(%6 : memref<64x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>>) {
  ^bb0(%arg4: f32, %arg5: f32):
    linalg.yield %arg4 : f32
  }
  func.return
}

// CHECK-LABEL: @fill2d
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %arg0
//       CHECK: vmvx.fill2d scalar(%arg1 : f32) out(%[[BB0]] offset %[[OFFSET0]] row_stride %[[STRIDES0]]#0 : !util.buffer) sizes(%[[SIZES0]]#0, %[[SIZES0]]#1)
func.func @fill2d(%arg0 : memref<384x128xf32>, %arg1 : f32) {
  linalg.fill ins(%arg1 : f32) outs(%arg0 : memref<384x128xf32>)
  func.return
}

// CHECK-LABEL: @matmul_f32f32f32_row_major
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %arg0
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZES1:.*]]:2, %[[STRIDES1:.*]]:2 = vmvx.get_buffer_descriptor %arg1
//   CHECK-DAG: %[[BB2:.*]], %[[OFFSET2:.*]], %[[SIZES2:.*]]:2, %[[STRIDES2:.*]]:2 = vmvx.get_buffer_descriptor %arg2
//       CHECK: vmvx.matmul lhs(%[[BB1]] offset %[[OFFSET1]] row_stride %[[STRIDES1]]#0 : !util.buffer)
//  CHECK-SAME:   rhs(%[[BB2]] offset %[[OFFSET2]] row_stride %[[STRIDES2]]#0 : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] row_stride %[[STRIDES0]]#0 : !util.buffer)
//  CHECK-SAME:   mnk(%[[SIZES1]]#0, %[[SIZES2]]#1, %[[SIZES2]]#0)
//  CHECK-SAME:   flags(1)
func.func @matmul_f32f32f32_row_major(%arg0 : memref<64x64xf32>, %arg1 : memref<64x384xf32>, %arg2 : memref<384x64xf32>) {
  linalg.matmul
      ins(%arg1, %arg2 : memref<64x384xf32>, memref<384x64xf32>)
      outs(%arg0 : memref<64x64xf32>)
  func.return
}

// CHECK-LABEL: @matmul_i8i8i32_row_major
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %arg0
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZES1:.*]]:2, %[[STRIDES1:.*]]:2 = vmvx.get_buffer_descriptor %arg1
//   CHECK-DAG: %[[BB2:.*]], %[[OFFSET2:.*]], %[[SIZES2:.*]]:2, %[[STRIDES2:.*]]:2 = vmvx.get_buffer_descriptor %arg2
//       CHECK: vmvx.matmul lhs(%[[BB1]] offset %[[OFFSET1]] row_stride %[[STRIDES1]]#0 : !util.buffer)
//  CHECK-SAME:   rhs(%[[BB2]] offset %[[OFFSET2]] row_stride %[[STRIDES2]]#0 : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] row_stride %[[STRIDES0]]#0 : !util.buffer)
//  CHECK-SAME:   mnk(%[[SIZES1]]#0, %[[SIZES2]]#1, %[[SIZES2]]#0)
//  CHECK-SAME:   flags(1)
func.func @matmul_i8i8i32_row_major(%arg0 : memref<64x64xi32>, %arg1 : memref<64x384xi8>, %arg2 : memref<384x64xi8>) {
  linalg.matmul
      ins(%arg1, %arg2 : memref<64x384xi8>, memref<384x64xi8>)
      outs(%arg0 : memref<64x64xi32>)
  func.return
}

// CHECK-LABEL: @matmul_i8i8i32_row_major_clear
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %arg0
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZES1:.*]]:2, %[[STRIDES1:.*]]:2 = vmvx.get_buffer_descriptor %arg1
//   CHECK-DAG: %[[BB2:.*]], %[[OFFSET2:.*]], %[[SIZES2:.*]]:2, %[[STRIDES2:.*]]:2 = vmvx.get_buffer_descriptor %arg2
//  CHECK-NEXT: vmvx.matmul lhs(%[[BB1]] offset %[[OFFSET1]] row_stride %[[STRIDES1]]#0 : !util.buffer)
//  CHECK-SAME:   rhs(%[[BB2]] offset %[[OFFSET2]] row_stride %[[STRIDES2]]#0 : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] row_stride %[[STRIDES0]]#0 : !util.buffer)
//  CHECK-SAME:   mnk(%[[SIZES1]]#0, %[[SIZES2]]#1, %[[SIZES2]]#0)
//  CHECK-SAME:   flags(0)
func.func @matmul_i8i8i32_row_major_clear(%arg0 : memref<64x64xi32>, %arg1 : memref<64x384xi8>, %arg2 : memref<384x64xi8>) {
  %c0 = arith.constant 0 : i32
  linalg.fill ins(%c0 : i32) outs(%arg0 : memref<64x64xi32>)
  linalg.matmul
      ins(%arg1, %arg2 : memref<64x384xi8>, memref<384x64xi8>)
      outs(%arg0 : memref<64x64xi32>)
  func.return
}

// CHECK-LABEL: @addf2d_rank_broadcast
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZES0:.*]]:2, %[[STRIDES0:.*]]:2 = vmvx.get_buffer_descriptor %arg0
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]], %[[SIZE1:.*]], %[[STRIDE1:.*]] = vmvx.get_buffer_descriptor %arg1
//       CHECK: vmvx.binary op("add" : f32) lhs(%[[BB1]] offset %[[OFFSET1]] strides[%[[C0]], %[[STRIDE1]]] : !util.buffer)
//  CHECK-SAME:   rhs(%[[BB0]] offset %[[OFFSET0]] strides[%[[STRIDES0]]#0, %[[STRIDES0]]#1] : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] strides[%[[STRIDES0]]#0, %[[STRIDES0]]#1] : !util.buffer)
//  CHECK-SAME:   sizes(%[[SIZES0]]#0, %[[SIZES0]]#1)
func.func @addf2d_rank_broadcast(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.addf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @addf0d
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[BB0:.*]], %[[OFFSET0:.*]], %[[SIZE0:.*]], %[[STRIDE0:.*]] = vmvx.get_buffer_descriptor %arg0
//   CHECK-DAG: %[[BB1:.*]], %[[OFFSET1:.*]] = vmvx.get_buffer_descriptor %arg1
//       CHECK: vmvx.binary op("add" : f32) lhs(%[[BB1]] offset %[[OFFSET1]] strides[%[[C0]], %[[C0]]] : !util.buffer)
//  CHECK-SAME:   rhs(%[[BB0]] offset %[[OFFSET0]] strides[%[[C0]], %[[STRIDE0]]] : !util.buffer)
//  CHECK-SAME:   out(%[[BB0]] offset %[[OFFSET0]] strides[%[[C0]], %[[STRIDE0]]] : !util.buffer) sizes(%[[C1]], %[[SIZE0]])
func.func @addf0d(%arg0 : memref<2xf32>, %arg1 : memref<f32>) {
  linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
    ins(%arg1 : memref<f32>) outs(%arg0 : memref<2xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.addf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert. Split by
// type because it is easier to copy/paste.
// CHECK-LABEL: @addi
// CHECK: vmvx.binary op("add" : i32)
func.func @addi(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.addi %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @andi
// CHECK: vmvx.binary op("and" : i32)
func.func @andi(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.andi %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @divsi
// CHECK: vmvx.binary op("divs" : i32)
func.func @divsi(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.divsi %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @divui
// CHECK: vmvx.binary op("divu" : i32)
func.func @divui(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.divui %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @muli
// CHECK: vmvx.binary op("mul" : i32)
func.func @muli(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.muli %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @ori
// CHECK: vmvx.binary op("or" : i32)
func.func @ori(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.ori %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @shli
// CHECK: vmvx.binary op("shl" : i32)
func.func @shli(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.shli %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @shrsi
// CHECK: vmvx.binary op("shrs" : i32)
func.func @shrsi(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.shrsi %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @xori
// CHECK: vmvx.binary op("xor" : i32)
func.func @xori(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.xori %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// Now test all binary primitives just to make sure they convert.
// CHECK-LABEL: @subi
// CHECK: vmvx.binary op("sub" : i32)
func.func @subi(%arg0 : memref<64x64xi32>, %arg1 : memref<64xi32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xi32>) outs(%arg0 : memref<64x64xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):
    %12 = arith.subi %arg2, %arg3 : i32
    linalg.yield %12 : i32
  }
  func.return
}

// CHECK-LABEL: @divf
// CHECK: vmvx.binary op("div" : f32)
func.func @divf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.divf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @mulf
// CHECK: vmvx.binary op("mul" : f32)
func.func @mulf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.mulf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @subf
// CHECK: vmvx.binary op("sub" : f32)
func.func @subf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.subf %arg2, %arg3 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// Unary ops.
// CHECK-LABEL: @absf
// CHECK: vmvx.unary op("abs" : f32)
func.func @absf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.absf %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @ceilf
// CHECK: vmvx.unary op("ceil" : f32)
func.func @ceilf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.ceil %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @exp
// CHECK: vmvx.unary op("exp" : f32)
func.func @expf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.exp %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @floorf
// CHECK: vmvx.unary op("floor" : f32)
func.func @floorf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.floor %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @log
// CHECK: vmvx.unary op("log" : f32)
func.func @logf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.log %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @negf
// CHECK: vmvx.unary op("neg" : f32)
func.func @negf(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = arith.negf %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}

// CHECK-LABEL: @rsqrt
// CHECK: vmvx.unary op("rsqrt" : f32)
func.func @rsqrt(%arg0 : memref<64x64xf32>, %arg1 : memref<64xf32>) {
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
    ins(%arg1 : memref<64xf32>) outs(%arg0 : memref<64x64xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %12 = math.rsqrt %arg2 : f32
    linalg.yield %12 : f32
  }
  func.return
}
