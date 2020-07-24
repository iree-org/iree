# Function Signatures

A key job of the IREE compiler and runtime is capturing function call semantics
from the originating system and providing mechanisms so that invocations can be
performed in as similar way as possible in various target languages. In general,
this requires additional metadata on top of the raw characteristics of a
function. Where possible, this is done by attaching attributes to a function.

*   `abi` : string indicating the abi/calling convention in use
*   `abiv` : numeric version of the `abi`

Each abi can require additional attributes as needed.

## Generic Signature Mangling

Where possible, ABI metadata is encoded into a plain-text signature in a way
that is easily transported across component boundaries and can be efficiently
implemented without additional dependencies (i.e. just string manipulation).

The suggested format is manipulated via the C++ reference implementations
`SignatureBuilder` and `SignatureParser` classes (see
`iree/base/signature_mangle.h`). See documentation and code for those classes
for more details.

## ABIs

### Raw Function ABI

All exported functions implement the raw function ABI, which defines the
metadata and calling convention for marshalling inputs and results to their
underlying implementations.

*Attributes:*

*   `fv` = 1 (current version of the raw function ABI)
*   `f` = encoded raw function signature (see below)
*   `fbr` = result buffer allocation function name (optional)

The reflection metadata documented here augments the underlying type system such
that host language bindings can interop as needed. This additional metadata is
needed in most dynamic cases because the compiled assets operate on fundamental
types with most characteristics type erased away (think: `void*` level things vs
high-level `ShapedBuffer` level things).

#### Grammar

The signature is implemented in terms of the SignatureBuilder, using tagged
Integer and Spans.

```text
signature ::= 'I' length-prefixed(type-sequence)
              'R' length-prefixed(type-sequence)

type-sequence ::= (arg-result-type)*
arg-result-type ::= buffer-type
                  | ref-object-type
                  | scalar-type
                  | unrecognized-type
buffer-type ::= 'B' length-prefixed(scalar-element-type? dim*)
scalar-type ::= 'S' length-prefixed(scalar-element-type?)
scalar-element-type ::= 't' (
                    '0'  # IEEE float32 (default if not specified)
                  | '1'  # IEEE float16
                  | '2'  # IEEE float64
                  | '3'  # Google bfloat16
                  | '4'  # Signed int8
                  | '5'  # Signed int16
                  | '6'  # Signed int32
                  | '7'  # Signed int64
                  | '8'  # Unsigned int8
                  | '9'  # Unsigned int16
                  | '10' # Unsigned int32
                  | '11' # Unsigned int64
                  )
dim :: = 'd' integer  # -1 indicates a dynamic dim
ref-object-type ::= 'O' length-prefixed()  # Details TBD
unrecognized-type ::= 'U' length-prefixed()

# Lexical primitives
integer ::= -?[0-9]+
length ::= [0-9]+
# The `length` encodes the length in bytes of `production`, plus 1 for the '!'.
length-prefixed(production) ::= length '!' production
any-byte-sequence ::= <any byte sequence>
```

#### Interpretation and Rationale

##### Memory layout

The astute reader will note that the above metadata is insufficient to determine
the memory layout of a buffer. The reason is that any more specific details than
this (contiguity, strides, alignment, etc) can actually only be known once the
actual compute devices have been enumerated and the resulting matrix of
conversions is more dynamic than can be expressed in something as static as a
function signature. The above formulation is an input to an additional runtime
oracle which produces appropriate full buffer descriptions.

While the exact implementation is host-language specific, consider the following
more detailed set of declarations that may exist in such a binding layer:

```c++
// Inspired heavily by the Py_buffer type.
// See: https://docs.python.org/3/c-api/buffer.html
struct BufferDescription {
  ScalarType element_type;
  // For contiguous arrays, this is is the length of the underlying memory.
  // For non-contiguous, this is the size of the buffer if it were copied
  // to a contiguous representation.
  size_t len;
  // Number of dims and strides.
  size_t ndim;
  int* shape;
  int* strides;
};

// Mirrors the 'buffer-type' production in the above grammar.
struct SignatureBufferType;

// Oracle which combines signature metadata with a user-provided, materialized
// BufferDescription to derive a BufferDescription that is compatible for
// invocation. Returns an updated buffer description if the original is
// not compatible or fully specified.
// This can be used in a couple of ways:
//   a) On function invocation to determine whether a provided buffer can be
//      used as-is or needs to be converted (copied).
//   b) To provide a factory function to the host language to create a
//      compatible buffer.
optional<BufferDescription> BufferDescriptionOracle(
    DeviceContext*, SignatureBufferType, BufferDescription)
  throws UnsupportedBufferException;
```

The above scheme should allow host-language and device coordination with respect
to buffer layout. For the moment, the responsibility to convert the buffer to a
compatible memory layout is on the host-language binding. However, often it is
the most efficient to schedule this for execution on a device. In the future, it
is anticipated that there will be a built-in pathway for scheduling such a
conversion (which would allow pipelining and offload of buffer conversions).

##### Deferred result allocation

In general, exported functions accept pre-allocated results that should be
mutated. For the simplest cases, such results can be `null` and retrieved upon
completion of the function. This, however, puts severe limitations on the
ability to pipeline. For fully specified signatures (no dynamic shapes), the
`BufferDescriptionOracle` and the signature is sufficient to pre-allocate
appropriate results, which allows chains of result-producing invocations to be
pipelined.

If, however, a `buffer-type` is not fully specified, the compiler may emit a
special *result allocator* function, which will be referenced in the `fbr`
attribute. Such a function would have a signature like this:

```c++
tuple<buffer> __allocate_results(tuple<int> dynamic_dims);
```

Such a function takes a tuple of all dynamic buffer dims in the function input
signature and returns a tuple of allocated buffers for each dynamic result. Note
that it may not be possible to fully allocate results in this fashion (i.e. if
the result layout is data dependent), in which case a null buffer is returned
for that slot (and the host library would need to await on the invocation to get
the fully populated result).

A similar mechanism will need to be created at some future point for
under-specified results of other (non-buffer) types.

##### Contiguity hinting

Commonly in some kinds of dataflows, the compiler needs to be free to internally
toggle buffer continuity (i.e. C/row-major, Fortran/col-major, etc). In many
cases, such toggling does not naturally escape through the exported function
boundaries, in which case, there is no ABI impact. However, it is anticipated
that there is benefit to letting the toggle propagate through the exported ABI
boundary, in which case, the `buffer-type` will likely be extended with a
contiguity hint indicating the preference. When combined with the buffer
description oracle and in-pipeline conversion features described above, this
could yield a powerful mechanism for dynamically and efficiently managing such
transitions.

Such an enhancement would almost certainly necessitate a major version bump in
the ABI and would be logical to implement once the advanced features above are
functional.

### Structured Index Path ABI

Functions may support the SIP ABI if their input and result tuples logically map
onto "structures" (nested sequence/dicts).

*Attributes:*

*   `sipv` = 1 (current SIP ABI version)
*   `sip` = encoded SIP signature (see below)

This ABI maps a raw, linear sequence of inputs and results onto an input and
result "structure" -- which in this context refers to a nested assembly of
sequences (with integer keys) and dictionaries (with string keys). Such a
facility is useful for encoding input/result mappings in a way that is common in
dynamic languages (such as Python).

In practice, this ABI supports the calling convention for TensorFlow, which
allows functions that accept and produce nestings via the
[`tf.nest`](https://www.tensorflow.org/api_docs/python/tf/nest) facility. In
implementing it, however, care has been taken to allow the calling convention to
generalize to other similar cases.

#### Grammar

The signature is implemented in terms of the SignatureBuilder, using tagged
Integer and Spans.

```text
# Defines the structured value for the inputs ('I') and results ('R')
# of the function.
signature ::= 'I' length-prefixed(structured-value)
              'R' length-prefixed(structured-value)

structured-value ::= raw-fn-index | sequence | dict
raw-fn-index ::= '_' integer
sequence ::= 'S' length-prefixed( (integer-key structured-value)* )
integer-key ::= 'k' integer
dict ::= 'D' length-prefixed( (string-key structured-value)* )
string-key ::= 'K' length-prefixed( any-byte-sequence )

# Low-level lexical primitives:
integer ::= -?[0-9]+
length ::= [0-9]+
# The `length` encodes the length in bytes of `production`, plus 1 for the '!'.
length-prefixed(production) ::= length '!' production
any-byte-sequence ::= <any byte sequence>
```

Structured values define a tree of recursive dicts/lists, with `raw-fn-index` at
the leaves. The interpretation is that a raw-fn-index that has been reached by
traversing N expansions of the structured-value production is assigned an "index
path" which is a list of the N keys that were traversed to reach it. For
example, for N=0, the index path is empty. For N=1, and if an integer-key with
numerical value 0 was traversed to reach the raw-fn-index, then the index path
is [0].

.... give a few examples more, writing out various nested dicts/lists in
Python-esque notation to clarify this concept ....

See the `SipSignatureParser::ToStringVisitor` for a canonical example of how to
interpret the signature.

#### Implementations

*   C++
    *   `SipSignatureMangler`: Produces a function signature given individual
        input and result assignment of physical indices to nested index paths in
        the structure tree.
    *   `SipSignatureParser`: Parses signatures and dispatches calls to a
        visitor.
