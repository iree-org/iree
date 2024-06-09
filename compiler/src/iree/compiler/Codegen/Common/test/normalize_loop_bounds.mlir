
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-codegen-normalize-loop-bounds, cse)" --allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s
module {
  func.func @for_normalize_step() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c0 to %c8 step %c2 {
      "iree.keep"(%arg0) : (index) -> ()
    }
    return
  }
}

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: func.func @for_normalize_step
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])

// -----

module {
  func.func @for_normalize_lowerbound() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    scf.for %arg0 = %c2 to %c8 step %c1 {
      "iree.keep"(%arg0) : (index) -> ()
    }
    return
  }
}

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL: func.func @for_normalize_lowerbound
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])

// -----

module {
  func.func @for_normalize_lowerbound_and_step() {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c13 = arith.constant 13 : index
    scf.for %arg0 = %c1 to %c13 step %c4 {
      "iree.keep"(%arg0) : (index) -> ()
    }
    return
  }
}

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 4 + 1)>
// CHECK-LABEL: func.func @for_normalize_lowerbound_and_step
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:       scf.for %[[ARG:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NEXT:    affine.apply #[[$MAP]](%[[ARG]])

// -----

module {
  func.func @forall_normalize_step() {
    scf.forall (%arg0, %arg1) = (0, 0) to (8, 16) step (8, 8) {
      "iree.keep"(%arg0, %arg1) : (index, index) -> ()
    }
    return
  }
}

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-LABEL: func.func @forall_normalize_step
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (1, 2)
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP]](%[[ARG1]])

// -----

module {
  func.func @forall_normalize_lowerbound() {
    scf.forall (%arg0, %arg1) = (2, 4) to (8, 16) step (1, 1) {
      "iree.keep"(%arg0, %arg1) : (index, index) -> ()
    }
    return
  }
}

// CHECK-DAG:   #[[$MAP0:.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL: func.func @forall_normalize_lowerbound
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (6, 12)
// CHECK-DAG:     affine.apply #[[$MAP1]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG1]])

// -----

module {
  func.func @forall_normalize_lowerbound_and_step() {
    scf.forall (%arg0, %arg1) = (2, 4) to (8, 16) step (2, 4) {
      "iree.keep"(%arg0, %arg1) : (index, index) -> ()
    }
    return
  }
}

// CHECK-DAG:   #[[$MAP0:.+]] = affine_map<(d0) -> (d0 * 4 + 4)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 2 + 2)>
// CHECK-LABEL: func.func @forall_normalize_lowerbound
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (3, 3)
// CHECK-DAG:     affine.apply #[[$MAP1]](%[[ARG0]])
// CHECK-DAG:     affine.apply #[[$MAP0]](%[[ARG1]])
