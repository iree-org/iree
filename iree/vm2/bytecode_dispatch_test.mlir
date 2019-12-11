// These test functions are called by the bytecode_dispatch_test.cc runner.
// The prefix of fail_ can be used to denote that the test is expected to fail
// (error returned from dispatch).
vm.module @bytecode_dispatch_test {
  // Tests that an empty function (0 args, 0 results, 0 ops) works.
  vm.export @empty
  vm.func @empty() {
    vm.return
  }

  // TODO(benvanik): more tests.
}
