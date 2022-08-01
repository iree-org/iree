# Custom modules in Python sample

This sample illustrates how to define a custom module in the Python API,
with a pure Python implementation, and compiling an overall program that
can use it.

This builds on the capabilities ot the `custom_module` sample, which
demonstrates C-based extension modules -- applying the same basics to
Python. Some features are not yet implemented on the Python side, and
the API is lower level than we should ultimately have. However, as
is demonstrated, it can do some not trivial things.

## Sample description

To show off some of the capabilities, this sample:

* Demonstrates how to define a custom Python function which accepts both
  a buffer and a variant list. Within the implementation, the buffer is
  wrapped by a numpy array for use.
* Module state is kept for the detokenizer state, keeping track of whether
  we are at the start of text or sentence. Real detokenizers are much
  more complex and would likely involve an opaque module custom type
  (not yet implemented in Python).
* A global in the main program is used to accumulate fragments by
  the `@detokenizer.accumtokens` function.
* The `@detokenizer.jointokens` will format and emit the text corresponding
  to accumulated tokens, respecting sentence boundaries and previous
  state.
* A `reset` function is exported which resets the accumulated tokens and
  the detokenizer state.

A real text model would be organized differently, but this example should
suffice to show that many of these advanced integration concepts are just
simple code.

A future version of this sample will embed the detokenizer vocabulary as
rodata in the main module and use that to initialize the internal lookup
table.