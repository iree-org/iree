# IREE Compiler C API

IREE exports three levels of APIs for integrating IREE (and uses these APIs
itself for building language bindings and binary tools):

* Low-level MLIR API: This API builds on top of the MLIR C-API in order to
  enable low level access to IR building, transformations and combined
  pipelines. It provides no defined ABI or API stability guarantees, although
  best effort is taken to not break the API unnecessarily (similar to how the
  upstream MLIR C-API is considered).
* Tool Entry Points: Entry points for standalone tools (i.e. `iree-compile`
  and `iree-lld`) are provided as exported C functions to enable the
  construction of busy-box binaries and other non-library use cases.
* Embedding API: The high level compiler API provides programmatic access to
  the facilities of `iree-compile` and is intended for embedding into an online
  system which needs to invoke the compiler as a library and is interopping
  via input and output artifacts (vs direct construction of IR against
  in-memory data structures). This API is versioned and both API/ABI compatible
  within a major version.

Depending on build configuration, the API is built in the following ways:

* Shared library: In this scenario, all of the APIs above are exported into
  a versioned shared library.
* Static library: Used for building a static tool (not enabled by default
  since it adds build overhead).
* Re-export Stub library: A static library which re-exports the Embedding API
  and provides additional entry-points for binding to a runtime-loaded
  shared library for the implementation. This is used as a convenience for
  consumers which need to decouple the interface from the implementation, which
  can happen in a variety of cases (i.e. avoiding symbol conflicts, integrating
  across build systems, etc).

See the implementation under `compiler/src/iree/compiler/API` (TODO: rename
to `ApiImpl`).