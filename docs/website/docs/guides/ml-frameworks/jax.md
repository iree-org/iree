---
hide:
  - tags
tags:
  - Python
  - JAX
icon: simple/python
---

# JAX integration

!!! note
    IREE's JAX support is under active development. This page is still under
    construction.

## :octicons-book-16: Overview

IREE offers two ways to interface with [JAX](https://github.com/google/jax)
programs:

* An API for extracting and compiling full models ahead of time (AOT) for
  execution apart from JAX. This API is being developed in the
  [iree-org/iree-jax repository](https://github.com/iree-org/iree-jax).
* A PJRT plugin that adapts IREE as a native JAX backend for online / just in
  time (JIT) use. This plugin is being developed in the
  [openxla/openxla-pjrt-plugin repository](https://github.com/openxla/openxla-pjrt-plugin).

<!-- TODO: Expand on interface differences -->
<!-- TODO: Add quickstart instructions -->
<!-- TODO: Link to samples -->
