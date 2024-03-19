// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-top-level-scf-to-cfg))" %s | FileCheck %s

// CHECK-LABEL: @generic_nested_for
// While not super recommended, we do have cases of SCF constructs embedded
// in linalg.generic. This sample was reduced from a lowering of tf.pow.
// The normal --convert-scf-to-std pass will produce an illegal linalg op
// (multiple basic blocks). The --iree-top-level-scf-to-cfg should not touch it.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
util.func public @generic_nested_for(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?x?x?x?xi32>, %out0: tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c-1_i32 = arith.constant -1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  // CHECK: linalg.generic
  // CHECK: scf.for
  // CHECK: linalg.yield
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>) outs(%out0 : tensor<?x?x?x?xi32>) {
  ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
    %18:3 = scf.for %arg5 = %c0 to %c6 step %c1 iter_args(%arg6 = %c1_i32, %arg7 = %arg2, %arg8 = %arg3) -> (i32, i32, i32) {
      %28 = arith.andi %arg8, %c1_i32 : i32
      %29 = arith.cmpi eq, %28, %c1_i32 : i32
      %30 = arith.muli %arg6, %arg7 : i32
      %31 = arith.select %29, %30, %arg6 : i32
      %32 = arith.muli %arg7, %arg7 : i32
      %33 = arith.shrui %arg8, %c1_i32 : i32
      scf.yield %31, %32, %33 : i32, i32, i32
    }
    %19 = arith.remsi %arg3, %c2_i32 : i32
    %20 = arith.cmpi eq, %19, %c0_i32 : i32
    %21 = arith.cmpi slt, %arg3, %c0_i32 : i32
    %22 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %23 = arith.cmpi eq, %arg2, %c-1_i32 : i32
    %24 = arith.select %22, %c1_i32, %c0_i32 : i32
    %25 = arith.select %20, %c1_i32, %c-1_i32 : i32
    %26 = arith.select %23, %25, %24 : i32
    %27 = arith.select %21, %26, %18#0 : i32
    linalg.yield %27 : i32
  } -> tensor<?x?x?x?xi32>

  util.return %0 : tensor<?x?x?x?xi32>
}
