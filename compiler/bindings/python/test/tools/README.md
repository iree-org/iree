# Python API Tests

These tests are run in an environment where all available Python bindings
are setup on the `PYTHONPATH`. Each will internally skip itself if optional
components are not available.

Note that IREE compiler tool locations can be overridden by specifying the
`IREE_TOOL_PATH` environment variable.
