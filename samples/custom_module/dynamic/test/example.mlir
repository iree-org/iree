// RUN: iree-compile %s --iree-hal-target-backends=vmvx | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=$IREE_BINARY_DIR/samples/custom_module/dynamic/module$IREE_DYLIB_EXT@create_custom_module \
// RUN:     --module=- \
// RUN:     --function=main | \
// RUN: FileCheck %s

// RUN: ( iree-compile %s --iree-hal-target-backends=vmvx | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=$IREE_BINARY_DIR/samples/custom_module/dynamic/module$IREE_DYLIB_EXT@create_custom_module \
// RUN:     --module=- \
// RUN:     --function=error 2>&1 || [[ $? == 1 ]] ) | \
// RUN: FileCheck %s --check-prefix=CERROR

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

  // Always returns unknown status with a custom annotation.
  func.func private @custom.error()

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

  // CERROR-LABEL: EXEC @error
  func.func @error() {
    // Show an example of emitting an error
    // CERROR-NEXT: UNKNOWN
    call @custom.error() : () -> ()
    return
  }

}
