# IREE Python Sandbox

This directory contains various integration-oriented Python utilities that are
not intended to be a public API. They are, however, useful for lower level
compiler interop work. And of course, they are useful since we presently lack a
real API :)

We're still untangling build support, jupyter integration, etc for OSS builds.
Stand by.

## Issues:

*   This is called `pyiree` vs `iree` to avoid pythonpath collisions that tend
    to arise when an iree directory is inside of an iree directory.
*   The above could be solved in the bazel build by making iree/bindings/python
    its own sub-workspace.
*   However, doing so presently breaks both flatbuffer and tablegen generation
    because of fixes needed to those build rules so that they are sub-worksapce
    aware.
