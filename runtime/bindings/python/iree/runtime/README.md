# IREE Python Runtime Components

This package provides an API for running compiled IREE binaries and interfacing with the hardware-abstraction-layer.

## Tracing

Execution of calls against binaries can be traced for later replay (i.e. via
tools like `iree-run-module`). This can be set up either explicitly or
via environment variables.

To trace via environment variable, set `IREE_SAVE_CALLS` to a directory to dump
traces into. Each created `SystemContext` will result in one `calls.yaml`
file (with an index appended to the stem for multiples). Any referenced
module binaries will be dumped into the same directory and referenced by the
YAML file.

### Explicit API

```python
tracer = iree.runtime.Tracer(some_dir)
config = iree.runtime.Config(driver, tracer)
...
```
