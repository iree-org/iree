# IREE TensorFlow Importers

This project contains IREE frontends for importing various forms of TensorFlow
formats.

## Quick Development Setup

This assumes that you have an appropriate `bazel` installed.

Build the importer binaries:

```
# All of them (takes a long time).
bazel build iree_tf_compiler:importer-binaries

# Or individuals:
bazel build iree_tf_compiler:iree-import-tflite
bazel build iree_tf_compiler:iree-import-xla
bazel build iree_tf_compiler:iree-import-tf
```

## Notes:

This directory is in a transitional state to its own project. Currently it
has its directory structure set up for that eventuality. Specifically,
`iree-dialects` is a symlink to the directory from the main repo. When split,
this will be a copy.