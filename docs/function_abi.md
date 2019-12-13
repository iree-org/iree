# Function signatures

A key job of the IREE compiler and runtime is capturing function call semantics
from the originating system and providing mechanisms so that invocations can be
performed in as similar way as possible in various target languages. In general,
this requires additional metadata on top of the raw characteristics of a
function. Where possible, this is done by attaching attributes to a function.

*   `abi` : string indiciating the abi/calling convention in use
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

### Structured Index Path ABI

*   `abi` = `sip`
*   Current `abiv` version = 1

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
