
//===----------------------------------------------------------------------===//
// b = add(a, a)
//===----------------------------------------------------------------------===//

// NOTE: this represents what our tflite import flow should produce; the _
// prefixed functions are all synthesized by us. We use the VM dialect in here
// now because std has no list and other stuff. In a real flow we may have a
// iree_tflite dialect that has pseudo ops for these things that then plug into
// the VM conversion interface, or maybe we just emit them as-is at input
// because we know where they are going to end up.

//===----------------------------------------------------------------------===//
// Global variable initialization
//===----------------------------------------------------------------------===//
// Optional generated function to reset any globals we may have in response to a
// TfLiteInterpreterResetVariableTensors call.

func @_reset_variables() {
  return
}
vm.export @_reset_variables

//===----------------------------------------------------------------------===//
// Shape I/O queries
//===----------------------------------------------------------------------===//
// Generated shape query functions so that we can preallocate (when possible).
// Note that if there are dynamic shapes some dimensions may be -1. By having
// these as functions we simplify the frontend (no need for signature parsing
// stuff) and allow all of this to be compiler-controlled even after deployment.
// For example, if we wanted to have these shapes based on the variable shapes
// or even previously computed results we could. The whole resizing operation
// becomes something we control in the compiler and not something we need the
// runtime to do for us (shape propagation/etc).

// TODO(#3973): allow index in vm.list so we can just convert based on
// the supported VM extension mode (i32 or i64).
func @_query_input_shape(%index : i32, %shape : !vm.list<i32>) {
  // NOTE: we could switch on %index but we only have one input.

  // TODO(#3973): improved ergonomics here; should have a variadic set.
  %rank = constant 4 : i32
  vm.list.resize %shape, %rank : (!vm.list<i32>, i32)
  %i0 = constant 0 : i32
  %i1 = constant 1 : i32
  %i2 = constant 2 : i32
  %i3 = constant 3 : i32
  %c1 = constant 1 : i32
  %c3 = constant 3 : i32
  %c8 = constant 8 : i32
  vm.list.set.i32 %shape, %i0, %c1 : (!vm.list<i32>, i32, i32)
  vm.list.set.i32 %shape, %i1, %c8 : (!vm.list<i32>, i32, i32)
  vm.list.set.i32 %shape, %i2, %c8 : (!vm.list<i32>, i32, i32)
  vm.list.set.i32 %shape, %i3, %c3 : (!vm.list<i32>, i32, i32)

  return
}
vm.export @_query_input_shape

// Handles TfLiteInterpreterResizeInputTensor calls.
// It could set vm.global.i32s for the relevant output shapes such that queries
// there pick up the propagated values. Since this is all just code that's
// likely to end up as just a handful of muls/divs after CSE (in what would
// normally be a full "shape propagation" pass).
func @_resize_input_shape(%index : i32, %shape : !vm.list<i32>) {
  // Unused here (but also not yet implemented #3975 :).
  return
}
vm.export @_resize_input_shape

func @_query_output_shape(%index : i32, %shape : !vm.list<i32>) {
  // Same input/output in this particular example. If we weren't writing this by
  // hand and instead generated this then symbol deduping would help turn them
  // both into the same thing.
  call @_query_input_shape(%index, %shape) : (i32, !vm.list<i32>) -> ()
  return
}
vm.export @_query_output_shape

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//
// Only one entry point today, but multiple are supported in IREE. It's also
// possible to call functions, so a single _main could fork out based on input
// arguments to other functions if we wanted to preserve the tflite API while
// allowing more complex models.

// TODO(#3972): handle tagging !quant.uniform as attributes.
// TODO(#3974): handle propagating tf.entry_function tensor name attrs .
func @_main(
    %input : tensor<1x8x8x3xf32>
  ) -> tensor<1x8x8x3xf32> {
  %result = mhlo.add %input, %input : tensor<1x8x8x3xf32>
  return %result : tensor<1x8x8x3xf32>
}
vm.export @_main
