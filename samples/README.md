# IREE Samples

Also see the [iree-samples](https://github.com/google/iree-samples) repository.

Note that for samples which include C/C++ code, the samples/ directory itself
is added as an include directory if depending on the `defs` library.
As such, samples that include headers should include "iree" in their name in
some fashion so that paths are unique.