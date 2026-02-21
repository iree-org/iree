// RUN: echo "ok" | FileCheck %s --check-prefix=HDR
// RUN: echo "alpha" | FileCheck %s --check-prefix=ALPHA

// HDR: ok
// ALPHA: alpha

// -----

// CHECK-LABEL: @case1
// CASE1: hello
func @case1() {
}

// -----

// CHECK-LABEL: @case2
// Some comment
// CASE2: world
func @case2() {
}

// -----

// CHECK-LABEL: @case3
// CASE3: gamma
func @case3() {
}

// Additional scattered RUN lines that should be preserved and not interfere:
// RUN: echo "world" | FileCheck %s --check-prefix=NOTUSED
// NOTUSED: something-else-entirely
