// internGlobalScope test:
// Creates four globals, each with a #util.elements initial value.
// Adds an "iree.aliases" attribute indicating the index in the set of any
// global that aliases another initial value due to elements interning.
// RUN: iree-opt --pass-pipeline='builtin.module(iree-util-test-cas-resources{test-mode=internGlobalScope})' %s \
// RUN:   | FileCheck --check-prefix=CHECK_INTERN_GLOBAL_SCOPE %s
// CHECK_INTERN_GLOBAL_SCOPE: global public @test0 {iree.aliases = [3 : i32]}
// CHECK_INTERN_GLOBAL_SCOPE: global public @test1 {iree.aliases = []}
// CHECK_INTERN_GLOBAL_SCOPE: global public @test2 {iree.aliases = []}
// CHECK_INTERN_GLOBAL_SCOPE: global public @test3 {iree.aliases = [0 : i32]}

// internLocalScope test:
// Creates three resources:
//   @test0 (local) 1, 2, 3, 4
//   @test1 (local) 4, 3, 2, 1
//   @test2 (global) 4, 3, 2, 1
//   @test3 (local) 1, 2, 3, 4
// Resources 0, 3 should alias. 1 and 2 should not (since the local is created
// before the global and resolution does not work that way). This is a bit
// contrived but verifying the contract. Also resource 2 should have a
// duplicated hash, which we verify by checking for a "_1" suffix.
// RUN: iree-opt --pass-pipeline='builtin.module(iree-util-test-cas-resources{test-mode=internLocalScope})' %s \
// RUN:   | FileCheck --check-prefix=CHECK_INTERN_LOCAL_SCOPE %s
// CHECK_INTERN_LOCAL_SCOPE: global public @test0 {iree.aliases = [3 : i32]}
// CHECK_INTERN_LOCAL_SCOPE: global public @test1 {iree.aliases = []}
// CHECK_INTERN_LOCAL_SCOPE: global public @test2 {iree.aliases = []}
// CHECK_INTERN_LOCAL_SCOPE-SAME: _1
// CHECK_INTERN_LOCAL_SCOPE: global public @test3 {iree.aliases = [0 : i32]}

// internLocalInvalidate test:
// Creates three resources:
//   @test0 (local) 1, 2, 3, 4
//   @test1 (local) 4, 3, 2, 1
//   @test2 (global) 4, 3, 2, 1
//   @test3 (local) 1, 2, 3, 4
// The local scope is invalidated, retaining none. The ops are then marked with
// iree.resource-dead or iree.resource-live
// RUN: iree-opt --pass-pipeline='builtin.module(iree-util-test-cas-resources{test-mode=internLocalInvalidate})' %s \
// RUN:   | FileCheck --check-prefix=CHECK_LOCAL_INVALIDATE %s
// CHECK_LOCAL_INVALIDATE: global public @test0 {"iree.resource-dead"}
// CHECK_LOCAL_INVALIDATE: global public @test1 {"iree.resource-dead"}
// CHECK_LOCAL_INVALIDATE: global public @test2 {"iree.resource-live"}
// CHECK_LOCAL_INVALIDATE: global public @test3 {"iree.resource-dead"}

module {
}
