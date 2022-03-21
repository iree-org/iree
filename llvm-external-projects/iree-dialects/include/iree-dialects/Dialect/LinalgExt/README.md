This folder defines dialects, interfaces, operations and transformations that are
- experimental
- meant to eventually be upstreamed to LLVM.

These are used (or will be used) within IREE as and when required. They are not
meant to be part of "features" that IREE exposes, or part of IREEs public
API. Their use within IREE is an internal implementation detail.

Some of the transformations here might not be as well tested as others, mostly
depending on how load-bearing it is within IREE. Those that are heavily used are
expected to be well tested, but that might not be the case for experimental
features. They are expected to achieve the same level of fidelity and testing as
upstream MLIR when they are being transitioned out of IREE.
