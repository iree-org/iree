# IREE Compiler API Implementation

This directory contains the implementation for the publicly exported
C API. See the headers and stub implementation in `compiler/bindings/c`,
which are available for standalone use, regardless of whether the compiler
is built (i.e. can be used to dynamically bind to a shared library, etc).

## Exported symbols

See the script `generate_exports.py` which generates a number of checked
in files with names `api_exports.*` which are needed for various styles of
linking. Presently, this script scrapes a number of headers to generate an
export list, but this is a WIP: we would like to align upstream better and
be explicit.

It is typically pretty obvious to determine that an update is needed: libraries
will fail to build on missing symbols, or language bindings will fail to load
at runtime (complaining of missing symbols).
