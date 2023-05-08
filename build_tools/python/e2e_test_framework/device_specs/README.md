# IREE Benchmark Device Specs

This direcotry contains the specifications of all target devices we run
benchmarks. Definitions can be found in the `*_specs.py` files.

## Adding a new device spec

1.  Register a unique device spec ID in
    [build_tools/python/e2e_test_framework/unique_ids.py](/build_tools/python/e2e_test_framework/unique_ids.py).
2.  Define the new device spec in an existing python module or create a new one.
