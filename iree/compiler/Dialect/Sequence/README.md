The `sequence` dialect models iterable sequences of things, and high-level
operations on those sequences. The current form of this dialect is inspired by
the high-level sequence logic in TFF, the first intended consumer of this
dialect. In the fullness of time, the goal is to align this dialect with the
future needs of `tf.data` and possibly upstream it to MLIR.

At this point, the only type defined by this dialect is `sequence.of<T>`,
parameterized by the type `T` of sequence elements.

The only supported operation is `sequence.map` that applies a specified mapping
function point-wise.

In the immediate future, we'll add `sequence.sum` and `sequence.reduce`, and
beyond this, other operations and types, as needed for functional parity with
`tf.data`.

NOTE: This dialect is currently work-in-progress and not ready for consumption.
