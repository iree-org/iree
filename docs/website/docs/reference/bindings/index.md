# API bindings

API bindings allow for programmatic use of IREE's compiler and runtime
components. The core IREE project is written in C[^1], allowing for API bindings
to be written in a variety of other languages.

!!! question - "Something missing?"

    Want to use another language? Looking for something specific out of one of
    those already listed?

    We welcome discussions on our
    [communication channels](../../index.md#communication-channels) and
    contributions on [our GitHub page](https://github.com/openxla/iree)!

## Official API bindings

Members of the core project team and OpenXLA partners maintain these official
bindings:

Language | Compiler API? | Runtime API? | Published packages?
-------- | ------------ | ----------- | ------------------
[C/C++](#cc) | :white_check_mark: Supported | :white_check_mark: Supported | :x: Unsupported
[Python](#python) | :white_check_mark: Supported | :white_check_mark: Supported | :white_check_mark: Supported

### C/C++

See the [C API](./c-api.md) reference page.

### :simple-python: Python

See the [Python](./python.md) reference page.

## Unofficial and experimental API bindings

Members of our developer community have authored bindings using other languages:

Language | Compiler API? | Runtime API? | Published packages?
-------- | ------------ | ----------- | ------------------
[JavaScript](#javascript) | :grey_question: Experimental | :grey_question: Experimental | :x: Unsupported
[Java](#java) | :x: Unsupported | :grey_question: Experimental | :x: Unsupported
[Julia](#julia) | :grey_question: Experimental | :grey_question: Experimental | :x: Unsupported
[Rust](#rust) | :x: Unsupported | :grey_question: Experimental | :grey_question: Experimental

### :simple-javascript: JavaScript

* JavaScript bindings for WebAssembly and WebGPU are under development in IREE's
[`experimental/web/`](https://github.com/openxla/iree/tree/main/experimental/web)
directory.

### :fontawesome-brands-java: Java

* Java TFLite bindings were developed at one point in IREE's
[`runtime/bindings/tflite/java`](https://github.com/openxla/iree/tree/main/runtime/bindings/tflite/java)
directory.

### :simple-julia: Julia

* [Coil.jl](https://github.com/Pangoraw/Coil.jl) is an experimental package to
lower and execute Julia tensor operations to IREE.

### :simple-rust: Rust

* [iree-rs](https://github.com/SamKG/iree-rs) is
[a crate](https://crates.io/crates/iree-rs) containing rustic bindings for the
IREE runtime.

[^1]: with some C++ tools and utilities
