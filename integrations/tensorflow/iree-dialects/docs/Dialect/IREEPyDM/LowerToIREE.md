# Lowering IREEPyDM to IREE

At the top-level, a program expressed in IREEPyDM ("PyDM") can be lowered to
IREE via the overall conversion pass `-convert-iree-pydm-to-iree`; however,
additional steps are needed to link a runtime library, etc.

This document provides application notes for how the conversion is
implemented.

## Type Conversion

Generally, types in PyDM map 1:1 with IREE types in a way which facilitates
a direct conversion.

## `object` and `object<type>` records

Objects map to IREE variant lists (`!iree.list<?>`) of N elements, where the
elements are laid out as:

* [0]: Type Code - i32 representing the type code of the object.
* [1] (optional): Data corresponding to the type.
* [2] (optional): i64 TypeID - if the `id()` function has been evaluated for
  this object, then this element will contain the unique id.

### Type Codes

The special type code `0` means "not assigned", which is distinct from `None`.
Other built-in types correspond to the `BuiltinTypeCode` enum in the dialect.

The following specifies the contents of the `data` field (1) of the object
record, per type code:

* `None`: No contents
* `Tuple`: An `!iree.list` of objects
* `List`: An `!iree.list` of objects
* `Str`: TODO: needs to be a union of i8/i16/i32 codepoint array
* `Bytes`: TODO: should be a memory buffer of some kind
* `ExceptionResult`: TODO: should hold a boxed exception of some kind
* `Type`: An object representing the type
* `Bool`: Implementation defined `iree_vm_value_type_t` field (typically
  `iree_vm_value_type_t.i8`).
* `Integer`: Implementation defined `iree_vm_value_type_t` field (typically
  `iree_vm_value_type_t.i32`).
* `Real`: Implementation defined `iree_vm_value_type_t` field (typically
  `iree_vm_value_type_t.f32`).
* `Complex`: TODO
* `Integer1` / `UInteger1`: `iree_vm_value_type_t.i8`
* `Integer2` / `UInteger2`: `iree_vm_value_type_t.i16`
* `Integer4` / `UInteger4`: `iree_vm_value_type_t.i32
* `Integer8` / `UInteger8`: `iree_vm_value_type_t.i64
* `Float2` : `iree_vm_value_type_t.i16` with bit value of an IEEE FP16
* `Float4` : `iree_vm_value_type_t.f32`
* `Float8` : `iree_vm_value_type_t.f64`
* `BFloat2` : `iree_vm_value_type_t.i16` with bit value of a BFloat16 (thanks
  Google).
* `Complex4` : TODO
* `Complex8` : TODO
* `Object` (and custom): An object record, as described here

### Mutability

The VM enforces no immutability constraints on the object records here (i.e.
they are just lists). If the compiler believes that it can recycle an object,
it is free to resize it appropriately and re-assign its contents. In this
way, things like closure cells and free variable cells can just be
allocated object records that that compiler completely reassigns the contents
of on assignment.

## Unboxed values

All values can exist in the program unboxed and will have a VM IR type which
corresponds to their runtime variant form (i.e. `!iree.list`, `i32`, etc).

## `!iree_pydm.exception_result`

The `exception_result` type is returned from every PyDM function indicating
whether the call completed successfully or in an error state. We encode it as
a signed `i32` value at runtime, where:

* `0` : indicates normal termination.
* `<0` : signals one of a predefined list of primitive exception types
  which do not contain any detail (i.e. `ValueException` analog).
* `>0` : is an index into a global exception lookup table with slots
  corresponding to logical threads of execution.

### Pre-defined exception codes:

* `-1` : `StopIteration` - Standard exception which indicates exhaustion of
  an iterator.
* `-2` : `StopAsyncIteration` - Standard exception which indicates exhaustion
  of an async iterator.
* `-3` : `RuntimeError` - general catch-all failure exception class.
* `-4` : `ValueError`
* `-5` : `NotImplementedError`
* `-6` : `KeyError`
* `-7` : `IndexError`
* `-8` : `AttributeError`
* `-9` : `TypeError`
* `-10` : `UnboundLocalError`

These pre-defined exception codes are allocated when the compiler has a need
to raise or interoperate with an exception category.
