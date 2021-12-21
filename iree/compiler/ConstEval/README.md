# JitEval

This directory contains compiler-in-compiler JIT evaluation tools for cases
where we want to delegate some portion of compilation to the compiler and
runtime itself. As this is a relatively fragile layering which depends on
~everything, these capabilities are isolated to this directory and they must
be configured to be used from top-level drivers in a way that is isolated and
optional from the perspective of the rest of the compiler.
