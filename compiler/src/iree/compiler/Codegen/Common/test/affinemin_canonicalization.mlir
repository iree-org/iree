// RUN: iree-opt --iree-codegen-affinemin-scf-canonicalization -canonicalize %s | FileCheck %s

// CHECK-LABEL: scf_for_distributed
func.func @scf_for_distributed(%A : memref<i64>, %id1 : index, %count1 : index,
                      %id2 : index, %count2 : index) {
  %c1020 = arith.constant 1020 : index
  %c1024 = arith.constant 1024 : index
  %0 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%id1]
  %1 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%count1]

  //  CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
  //  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i64
  //      CHECK: scf.for
  //      CHECK:   scf.for %{{.*}} = %{{.*}} to %[[C32]]
  // CHECK-NEXT:     memref.store %[[C4]], %{{.*}}[] : memref<i64>
  scf.for %arg0 = %0 to %c1024 step %1 {
    %2 = affine.min affine_map<(d0) -> (32, -d0 + 1024)>(%arg0)
    %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%id2]
    %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%count2]
    scf.for %arg1 = %3 to %2 step %4 {
      %5 = affine.min affine_map<(d0, d1) -> (4, d0 - d1)>(%2, %arg1)
      %6 = arith.index_cast %5: index to i64
      memref.store %6, %A[]: memref<i64>
    }
  }

  // In this case the first affine.min cannot be removed as the 1020 is not
  // divisible by 32 but we can still remove the second level of affine.min
  // since we know that %2 is always divisible by 4.
  //      CHECK: scf.for
  //      CHECK:   %[[MIN:.*]] = affine.min
  //      CHECK:   scf.for %{{.*}} = %{{.*}} to %[[MIN]]
  // CHECK-NEXT:     memref.store %[[C4]], %{{.*}}[] : memref<i64>
  scf.for %arg0 = %0 to %c1020 step %1 {
    %2 = affine.min affine_map<(d0) -> (32, -d0 + 1020)>(%arg0)
    %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%id2]
    %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%count2]
    scf.for %arg1 = %3 to %2 step %4 {
      %5 = affine.min affine_map<(d0, d1) -> (4, d0 - d1)>(%2, %arg1)
      %6 = arith.index_cast %5: index to i64
      memref.store %6, %A[]: memref<i64>
    }
  }

  // Same case but using scf.parallel.
  //      CHECK: scf.for
  //      CHECK:   %[[MIN:.*]] = affine.min
  //      CHECK:   scf.parallel {{.*}} to (%[[MIN]])
  // CHECK-NEXT:     memref.store %[[C4]], %{{.*}}[] : memref<i64>
  scf.for %arg0 = %0 to %c1020 step %1 {
    %2 = affine.min affine_map<(d0) -> (32, -d0 + 1020)>(%arg0)
    %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%id2]
    %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%count2]
    scf.parallel (%arg1) = (%3) to (%2) step (%4) {
      %5 = affine.min affine_map<(d0, d1) -> (4, d0 - d1)>(%2, %arg1)
      %6 = arith.index_cast %5: index to i64
      memref.store %6, %A[]: memref<i64>
    }
  }

  return
}
