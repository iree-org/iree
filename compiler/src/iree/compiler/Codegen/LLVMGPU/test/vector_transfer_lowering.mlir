// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-vector-transfer-lowering))" --split-input-file %s | FileCheck %s

func.func @broadcast_read_lowering(%arg0: memref<4096x32xf16>) -> vector<1x8xf16> {
  %cst = arith.constant 0.000000e+00 : f16
  %thread_id_x = gpu.thread_id x
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = vector.transfer_read %arg0[%workgroup_id_x, %thread_id_x], %cst {in_bounds = [true]} : memref<4096x32xf16>, vector<1xf16>
  %1 = vector.extract %0[0] : f16 from vector<1xf16>
  %2 = vector.broadcast %1 : f16 to vector<1x8xf16>
  return %2 : vector<1x8xf16>
}

// CHECK-LABEL: func.func @broadcast_read_lowering
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x32xf16>)
//  CHECK: %[[LOAD:.+]] = vector.load %[[ARG0]]{{.*}} : memref<4096x32xf16>
//  CHECK: %[[ELEM:.+]] = vector.extract %[[LOAD]][0] : f16 from vector<1xf16>
//  CHECK: %[[INSERT:.+]] = vector.broadcast %[[ELEM]] : f16 to vector<1x8xf16>
//  CHECK: return %[[INSERT]]

// -----

func.func @transfer_gather_unroll_embedding_lookup(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>) -> (vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = ub.poison : vector<64xf16>
  %1 = ub.poison : vector<64xf16>
  %2 = ub.poison : vector<64xf16>
  %3 = ub.poison : vector<64xf16>
  %4 = vector.extract %arg1[0] : index from vector<4xindex>
  %5 = vector.transfer_read %arg0[%4, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %6 = vector.extract %arg1[1] : index from vector<4xindex>
  %7 = vector.transfer_read %arg0[%6, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %8 = vector.extract %arg1[2] : index from vector<4xindex>
  %9 = vector.transfer_read %arg0[%8, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %10 = vector.extract %arg1[3] : index from vector<4xindex>
  %11 = vector.transfer_read %arg0[%10, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %5, %7, %9, %11 : vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>
}

// After unrolling + canonicalization, the 2D gather becomes 4 contiguous loads.
// CHECK-LABEL: func.func @transfer_gather_unroll_embedding_lookup
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.load
// CHECK-NOT: transfer_gather

// -----

func.func @transfer_gather_unroll_masked(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>, %arg2: vector<64xi1>, %arg3: vector<64xi1>, %arg4: vector<64xi1>, %arg5: vector<64xi1>) -> (vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = ub.poison : vector<64xf16>
  %1 = ub.poison : vector<64xf16>
  %2 = ub.poison : vector<64xf16>
  %3 = ub.poison : vector<64xf16>
  %4 = vector.extract %arg1[0] : index from vector<4xindex>
  %5 = vector.transfer_read %arg0[%4, %c0], %cst, %arg2 {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %6 = vector.extract %arg1[1] : index from vector<4xindex>
  %7 = vector.transfer_read %arg0[%6, %c0], %cst, %arg3 {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %8 = vector.extract %arg1[2] : index from vector<4xindex>
  %9 = vector.transfer_read %arg0[%8, %c0], %cst, %arg4 {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %10 = vector.extract %arg1[3] : index from vector<4xindex>
  %11 = vector.transfer_read %arg0[%10, %c0], %cst, %arg5 {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %5, %7, %9, %11 : vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>
}

// After unrolling, mask slices are passed to each sub-gather.
// The masked rank-1 gathers lower to vector.maskedload ops.
// CHECK-LABEL: func.func @transfer_gather_unroll_masked
// CHECK-NOT: transfer_gather
// CHECK-COUNT-4: vector.maskedload
// CHECK-NOT: transfer_gather

// -----

func.func @transfer_gather_unroll_transposed_index(%arg0: memref<4096x64xf16>, %arg1: vector<4xindex>, %arg2: vector<4xindex>, %arg3: vector<4xindex>, %arg4: vector<4xindex>, %arg5: vector<4xindex>, %arg6: vector<4xindex>, %arg7: vector<4xindex>, %arg8: vector<4xindex>) -> (vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = ub.poison : vector<64xf16>
  %1 = ub.poison : vector<64xf16>
  %2 = ub.poison : vector<64xf16>
  %3 = ub.poison : vector<64xf16>
  %4 = ub.poison : vector<64xf16>
  %5 = ub.poison : vector<64xf16>
  %6 = ub.poison : vector<64xf16>
  %7 = ub.poison : vector<64xf16>
  %8 = ub.poison : vector<64xf16>
  %9 = ub.poison : vector<64xf16>
  %10 = ub.poison : vector<64xf16>
  %11 = ub.poison : vector<64xf16>
  %12 = ub.poison : vector<64xf16>
  %13 = ub.poison : vector<64xf16>
  %14 = ub.poison : vector<64xf16>
  %15 = ub.poison : vector<64xf16>
  %16 = ub.poison : vector<64xf16>
  %17 = ub.poison : vector<64xf16>
  %18 = ub.poison : vector<64xf16>
  %19 = ub.poison : vector<64xf16>
  %20 = ub.poison : vector<64xf16>
  %21 = ub.poison : vector<64xf16>
  %22 = ub.poison : vector<64xf16>
  %23 = ub.poison : vector<64xf16>
  %24 = ub.poison : vector<64xf16>
  %25 = ub.poison : vector<64xf16>
  %26 = ub.poison : vector<64xf16>
  %27 = ub.poison : vector<64xf16>
  %28 = ub.poison : vector<64xf16>
  %29 = ub.poison : vector<64xf16>
  %30 = ub.poison : vector<64xf16>
  %31 = ub.poison : vector<64xf16>
  %32 = ub.poison : vector<64xf16>
  %33 = ub.poison : vector<64xf16>
  %34 = ub.poison : vector<64xf16>
  %35 = ub.poison : vector<64xf16>
  %36 = ub.poison : vector<64xf16>
  %37 = ub.poison : vector<64xf16>
  %38 = ub.poison : vector<64xf16>
  %39 = ub.poison : vector<64xf16>
  %40 = vector.extract %arg1[0] : index from vector<4xindex>
  %41 = vector.transfer_read %arg0[%40, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %42 = vector.extract %arg2[0] : index from vector<4xindex>
  %43 = vector.transfer_read %arg0[%42, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %44 = vector.extract %arg3[0] : index from vector<4xindex>
  %45 = vector.transfer_read %arg0[%44, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %46 = vector.extract %arg4[0] : index from vector<4xindex>
  %47 = vector.transfer_read %arg0[%46, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %48 = vector.extract %arg5[0] : index from vector<4xindex>
  %49 = vector.transfer_read %arg0[%48, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %50 = vector.extract %arg6[0] : index from vector<4xindex>
  %51 = vector.transfer_read %arg0[%50, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %52 = vector.extract %arg7[0] : index from vector<4xindex>
  %53 = vector.transfer_read %arg0[%52, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %54 = vector.extract %arg8[0] : index from vector<4xindex>
  %55 = vector.transfer_read %arg0[%54, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %56 = vector.extract %arg1[1] : index from vector<4xindex>
  %57 = vector.transfer_read %arg0[%56, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %58 = vector.extract %arg2[1] : index from vector<4xindex>
  %59 = vector.transfer_read %arg0[%58, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %60 = vector.extract %arg3[1] : index from vector<4xindex>
  %61 = vector.transfer_read %arg0[%60, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %62 = vector.extract %arg4[1] : index from vector<4xindex>
  %63 = vector.transfer_read %arg0[%62, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %64 = vector.extract %arg5[1] : index from vector<4xindex>
  %65 = vector.transfer_read %arg0[%64, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %66 = vector.extract %arg6[1] : index from vector<4xindex>
  %67 = vector.transfer_read %arg0[%66, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %68 = vector.extract %arg7[1] : index from vector<4xindex>
  %69 = vector.transfer_read %arg0[%68, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %70 = vector.extract %arg8[1] : index from vector<4xindex>
  %71 = vector.transfer_read %arg0[%70, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %72 = vector.extract %arg1[2] : index from vector<4xindex>
  %73 = vector.transfer_read %arg0[%72, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %74 = vector.extract %arg2[2] : index from vector<4xindex>
  %75 = vector.transfer_read %arg0[%74, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %76 = vector.extract %arg3[2] : index from vector<4xindex>
  %77 = vector.transfer_read %arg0[%76, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %78 = vector.extract %arg4[2] : index from vector<4xindex>
  %79 = vector.transfer_read %arg0[%78, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %80 = vector.extract %arg5[2] : index from vector<4xindex>
  %81 = vector.transfer_read %arg0[%80, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %82 = vector.extract %arg6[2] : index from vector<4xindex>
  %83 = vector.transfer_read %arg0[%82, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %84 = vector.extract %arg7[2] : index from vector<4xindex>
  %85 = vector.transfer_read %arg0[%84, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %86 = vector.extract %arg8[2] : index from vector<4xindex>
  %87 = vector.transfer_read %arg0[%86, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %88 = vector.extract %arg1[3] : index from vector<4xindex>
  %89 = vector.transfer_read %arg0[%88, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %90 = vector.extract %arg2[3] : index from vector<4xindex>
  %91 = vector.transfer_read %arg0[%90, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %92 = vector.extract %arg3[3] : index from vector<4xindex>
  %93 = vector.transfer_read %arg0[%92, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %94 = vector.extract %arg4[3] : index from vector<4xindex>
  %95 = vector.transfer_read %arg0[%94, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %96 = vector.extract %arg5[3] : index from vector<4xindex>
  %97 = vector.transfer_read %arg0[%96, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %98 = vector.extract %arg6[3] : index from vector<4xindex>
  %99 = vector.transfer_read %arg0[%98, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %100 = vector.extract %arg7[3] : index from vector<4xindex>
  %101 = vector.transfer_read %arg0[%100, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  %102 = vector.extract %arg8[3] : index from vector<4xindex>
  %103 = vector.transfer_read %arg0[%102, %c0], %cst {in_bounds = [true]} : memref<4096x64xf16>, vector<64xf16>
  return %41, %43, %45, %47, %49, %51, %53, %55, %57, %59, %61, %63, %65, %67, %69, %71, %73, %75, %77, %79, %81, %83, %85, %87, %89, %91, %93, %95, %97, %99, %101, %103 : vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>
}

// After two rounds of unrolling (d0=4 then d1=8) + canonicalization,
// the 3D gather becomes 4*8=32 contiguous loads.
// CHECK-LABEL: func.func @transfer_gather_unroll_transposed_index
// CHECK-NOT: transfer_gather
// CHECK-COUNT-32: vector.load
// CHECK-NOT: transfer_gather

// -----

func.func @transfer_scatter_unroll_embedding_write(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<64xf16>, %arg3: vector<64xf16>, %arg4: vector<64xf16>, %arg5: vector<4xindex>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg5[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %1 = vector.extract %arg5[1] : index from vector<4xindex>
  vector.transfer_write %arg2, %arg0[%1, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %2 = vector.extract %arg5[2] : index from vector<4xindex>
  vector.transfer_write %arg3, %arg0[%2, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %3 = vector.extract %arg5[3] : index from vector<4xindex>
  vector.transfer_write %arg4, %arg0[%3, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// After unrolling, the 2D scatter becomes 4 rank-1 stores.
// CHECK-LABEL: func.func @transfer_scatter_unroll_embedding_write
// CHECK-COUNT-4: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>

// -----

func.func @transfer_scatter_unroll_masked(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<64xf16>, %arg3: vector<64xf16>, %arg4: vector<64xf16>, %arg5: vector<4xindex>, %arg6: vector<64xi1>, %arg7: vector<64xi1>, %arg8: vector<64xi1>, %arg9: vector<64xi1>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg5[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0], %arg6 {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %1 = vector.extract %arg5[1] : index from vector<4xindex>
  vector.transfer_write %arg2, %arg0[%1, %c0], %arg7 {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %2 = vector.extract %arg5[2] : index from vector<4xindex>
  vector.transfer_write %arg3, %arg0[%2, %c0], %arg8 {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %3 = vector.extract %arg5[3] : index from vector<4xindex>
  vector.transfer_write %arg4, %arg0[%3, %c0], %arg9 {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// After unrolling, mask slices are passed to each masked store.
// CHECK-LABEL: func.func @transfer_scatter_unroll_masked
// CHECK-COUNT-4: vector.maskedstore {{.+}} : memref<4096x64xf16>, vector<64xi1>, vector<64xf16>

// -----

func.func @transfer_scatter_unroll_tensor(%arg0: tensor<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<64xf16>, %arg3: vector<64xf16>, %arg4: vector<64xf16>, %arg5: vector<4xindex>) -> tensor<4096x64xf16> {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg5[0] : index from vector<4xindex>
  %1 = vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<4096x64xf16>
  %2 = vector.extract %arg5[1] : index from vector<4xindex>
  %3 = vector.transfer_write %arg2, %1[%2, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<4096x64xf16>
  %4 = vector.extract %arg5[2] : index from vector<4xindex>
  %5 = vector.transfer_write %arg3, %3[%4, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<4096x64xf16>
  %6 = vector.extract %arg5[3] : index from vector<4xindex>
  %7 = vector.transfer_write %arg4, %5[%6, %c0] {in_bounds = [true]} : vector<64xf16>, tensor<4096x64xf16>
  return %7 : tensor<4096x64xf16>
}

// After unrolling, the 2D scatter becomes 4 rank-1 transfer_write chained
// via tensor SSA results.
// CHECK-LABEL: func.func @transfer_scatter_unroll_tensor
// CHECK-COUNT-4: vector.transfer_write {{.+}} : vector<64xf16>, tensor<4096x64xf16>

// -----

func.func @transfer_scatter_unroll_transposed_index(%arg0: memref<4096x64xf16>, %arg1: vector<64xf16>, %arg2: vector<64xf16>, %arg3: vector<64xf16>, %arg4: vector<64xf16>, %arg5: vector<64xf16>, %arg6: vector<64xf16>, %arg7: vector<64xf16>, %arg8: vector<64xf16>, %arg9: vector<64xf16>, %arg10: vector<64xf16>, %arg11: vector<64xf16>, %arg12: vector<64xf16>, %arg13: vector<64xf16>, %arg14: vector<64xf16>, %arg15: vector<64xf16>, %arg16: vector<64xf16>, %arg17: vector<64xf16>, %arg18: vector<64xf16>, %arg19: vector<64xf16>, %arg20: vector<64xf16>, %arg21: vector<64xf16>, %arg22: vector<64xf16>, %arg23: vector<64xf16>, %arg24: vector<64xf16>, %arg25: vector<64xf16>, %arg26: vector<64xf16>, %arg27: vector<64xf16>, %arg28: vector<64xf16>, %arg29: vector<64xf16>, %arg30: vector<64xf16>, %arg31: vector<64xf16>, %arg32: vector<64xf16>, %arg33: vector<4xindex>, %arg34: vector<4xindex>, %arg35: vector<4xindex>, %arg36: vector<4xindex>, %arg37: vector<4xindex>, %arg38: vector<4xindex>, %arg39: vector<4xindex>, %arg40: vector<4xindex>) {
  %c0 = arith.constant 0 : index
  %0 = vector.extract %arg33[0] : index from vector<4xindex>
  vector.transfer_write %arg1, %arg0[%0, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %1 = vector.extract %arg34[0] : index from vector<4xindex>
  vector.transfer_write %arg2, %arg0[%1, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %2 = vector.extract %arg35[0] : index from vector<4xindex>
  vector.transfer_write %arg3, %arg0[%2, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %3 = vector.extract %arg36[0] : index from vector<4xindex>
  vector.transfer_write %arg4, %arg0[%3, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %4 = vector.extract %arg37[0] : index from vector<4xindex>
  vector.transfer_write %arg5, %arg0[%4, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %5 = vector.extract %arg38[0] : index from vector<4xindex>
  vector.transfer_write %arg6, %arg0[%5, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %6 = vector.extract %arg39[0] : index from vector<4xindex>
  vector.transfer_write %arg7, %arg0[%6, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %7 = vector.extract %arg40[0] : index from vector<4xindex>
  vector.transfer_write %arg8, %arg0[%7, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %8 = vector.extract %arg33[1] : index from vector<4xindex>
  vector.transfer_write %arg9, %arg0[%8, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %9 = vector.extract %arg34[1] : index from vector<4xindex>
  vector.transfer_write %arg10, %arg0[%9, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %10 = vector.extract %arg35[1] : index from vector<4xindex>
  vector.transfer_write %arg11, %arg0[%10, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %11 = vector.extract %arg36[1] : index from vector<4xindex>
  vector.transfer_write %arg12, %arg0[%11, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %12 = vector.extract %arg37[1] : index from vector<4xindex>
  vector.transfer_write %arg13, %arg0[%12, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %13 = vector.extract %arg38[1] : index from vector<4xindex>
  vector.transfer_write %arg14, %arg0[%13, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %14 = vector.extract %arg39[1] : index from vector<4xindex>
  vector.transfer_write %arg15, %arg0[%14, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %15 = vector.extract %arg40[1] : index from vector<4xindex>
  vector.transfer_write %arg16, %arg0[%15, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %16 = vector.extract %arg33[2] : index from vector<4xindex>
  vector.transfer_write %arg17, %arg0[%16, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %17 = vector.extract %arg34[2] : index from vector<4xindex>
  vector.transfer_write %arg18, %arg0[%17, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %18 = vector.extract %arg35[2] : index from vector<4xindex>
  vector.transfer_write %arg19, %arg0[%18, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %19 = vector.extract %arg36[2] : index from vector<4xindex>
  vector.transfer_write %arg20, %arg0[%19, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %20 = vector.extract %arg37[2] : index from vector<4xindex>
  vector.transfer_write %arg21, %arg0[%20, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %21 = vector.extract %arg38[2] : index from vector<4xindex>
  vector.transfer_write %arg22, %arg0[%21, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %22 = vector.extract %arg39[2] : index from vector<4xindex>
  vector.transfer_write %arg23, %arg0[%22, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %23 = vector.extract %arg40[2] : index from vector<4xindex>
  vector.transfer_write %arg24, %arg0[%23, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %24 = vector.extract %arg33[3] : index from vector<4xindex>
  vector.transfer_write %arg25, %arg0[%24, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %25 = vector.extract %arg34[3] : index from vector<4xindex>
  vector.transfer_write %arg26, %arg0[%25, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %26 = vector.extract %arg35[3] : index from vector<4xindex>
  vector.transfer_write %arg27, %arg0[%26, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %27 = vector.extract %arg36[3] : index from vector<4xindex>
  vector.transfer_write %arg28, %arg0[%27, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %28 = vector.extract %arg37[3] : index from vector<4xindex>
  vector.transfer_write %arg29, %arg0[%28, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %29 = vector.extract %arg38[3] : index from vector<4xindex>
  vector.transfer_write %arg30, %arg0[%29, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %30 = vector.extract %arg39[3] : index from vector<4xindex>
  vector.transfer_write %arg31, %arg0[%30, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  %31 = vector.extract %arg40[3] : index from vector<4xindex>
  vector.transfer_write %arg32, %arg0[%31, %c0] {in_bounds = [true]} : vector<64xf16>, memref<4096x64xf16>
  return
}

// After two rounds of unrolling (d0=4 then d1=8), the 3D scatter
// becomes 4*8=32 rank-1 stores.
// CHECK-LABEL: func.func @transfer_scatter_unroll_transposed_index
// CHECK-COUNT-32: vector.store {{.+}} : memref<4096x64xf16>, vector<64xf16>
