// RUN: iree-opt --verify-diagnostics --iree-util-link-modules %s

// Tests error case: unresolved external symbol.

module {
  // expected-error @+1 {{unresolved external symbol: 'nonexistent.function'}}
  util.func private @nonexistent.function(%arg0: i32) -> i32

  util.func public @main(%arg0: i32) -> i32 {
    %0 = util.call @nonexistent.function(%arg0) : (i32) -> i32
    util.return %0 : i32
  }
}
