// RUN: iree-compile %s \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=vmvx | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=- \
// RUN:     --function=main | \
// RUN: FileCheck %s

module @example {
  //===--------------------------------------------------------------------===//
  // Imports
  //===--------------------------------------------------------------------===//
  // External function declarations for the methods implemented in the custom
  // module C++ file. Note that they are prefixed with the `custom.` module
  // name.

  // Creates a new string with contents from the given tensor.
  // This is silly. Don't do this :)
  func.func private @custom.string.from_tensor(tensor<?xi8>) -> !custom.string

  // Prints the contents of the string to stdout.
  func.func private @custom.string.print(!custom.string)

  //===--------------------------------------------------------------------===//
  // Sample methods
  //===--------------------------------------------------------------------===//
  // Note that there can be any number of publicly-exported methods; this simple
  // sample just has one to keep things simple.

  // CHECK-LABEL: EXEC @main
  func.func @main() {
    // Create string from a byte buffer encoding the characters.
    %hello_bytes = util.unfoldable_constant dense<[0, 1, 2, 3, 4]> : tensor<5xi8>
    %hello_arg = tensor.cast %hello_bytes : tensor<5xi8> to tensor<?xi8>
    // CHECK-NEXT: CREATE 5xi8=0 1 2 3 4
    %hello_str = call @custom.string.from_tensor(%hello_arg) : (tensor<?xi8>) -> !custom.string

    // Print the string to stdout.
    // CHECK-NEXT: PRINT 5xi8=0 1 2 3 4
    call @custom.string.print(%hello_str) : (!custom.string) -> ()

    return
  }
}
