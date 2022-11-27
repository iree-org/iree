// RUN: iree-compile %s --iree-execution-model=host-only | custom-module-basic-run - example.main | FileCheck %s

module @example {
  //===--------------------------------------------------------------------===//
  // Imports
  //===--------------------------------------------------------------------===//
  // External function declarations for the methods implemented in the custom
  // module C++ file. Note that they are prefixed with the `custom.` module
  // name.

  // Creates a new string with a copy of the given string data.
  // No NUL terminator is required.
  func.func private @custom.string.create(!util.buffer) -> !custom.string

  // Returns the length of the string in characters.
  func.func private @custom.string.length(!custom.string) -> index attributes {
    // Explicitly force the returned type to be i64 regardless of whether the
    // VM is in 32 or 64 bit mode and what conversion would make index.
    vm.signature = (!vm.ref<!custom.string>) -> i64
  }

  // Prints the contents of the string to stdout.
  func.func private @custom.string.print(!custom.string)

  // Prints the contents of the string only in debug mode and otherwise prints
  // "optimized".
  func.func private @custom.string.dprint(!custom.string) attributes {
    // Indicates the import is optional and if not present the specified
    // fallback method will be called instead.
    vm.fallback = @custom_string_dprint
  }
  func.func private @custom_string_dprint(%ignored: !custom.string) {
    // Called when the import is not available at runtime (in this case when the
    // runtime is compiled in release mode). This is a silly example but makes
    // it easier to test.
    %data = util.buffer.constant : !util.buffer = "optimized"
    %str = call @custom.string.create(%data) : (!util.buffer) -> !custom.string
    call @custom.string.print(%str) : (!custom.string) -> ()
    return
  }

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//
  // Note that there can be any number of publicly-exported methods; this simple
  // sample just has one to keep things simple.

  // CHECK-LABEL: INVOKE BEGIN example.main
  func.func @main() {
    // Create string from a byte buffer encoding the characters.
    %hello_data = util.buffer.constant : !util.buffer = "hello"
    // CHECK-NEXT: CREATE hello
    %hello_str = call @custom.string.create(%hello_data) : (!util.buffer) -> !custom.string

    // Print the string to stdout.
    // CHECK-NEXT: PRINT hello
    call @custom.string.print(%hello_str) : (!custom.string) -> ()

    // Query the length of the string.
    // We don't do anything with it here but just demonstrate how index works.
    // CHECK-NEXT: LENGTH hello = 5
    %strlen = call @custom.string.length(%hello_str) : (!custom.string) -> index
    util.optimization_barrier %strlen : index

    // Print "debug" if the runtime is compiled in debug mode and otherwise
    // prints "optimized".
    // CHECK: PRINT {{debug|optimized}}
    %debug_data = util.buffer.constant : !util.buffer = "debug"
    %debug_str = call @custom.string.create(%debug_data) : (!util.buffer) -> !custom.string
    call @custom.string.dprint(%debug_str) : (!custom.string) -> ()

    return
  }
  // CHECK-NEXT: INVOKE END
}
