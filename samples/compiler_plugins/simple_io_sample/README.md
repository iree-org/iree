# SimpleIO Compiler Plugin Sample

This sample demonstrates a compiler plugin which:

* Adds a new dialect to IREE
* Implements pre-processor lowerings to transform ops to internal
  implementations (TODO)
* Has a python-based runner that implements the IO ops in pure python (TODO)
* Illustrates some advanced features of the way such things can be
  constructed (custom types, async, etc).

To use this, the plugin must be built into the compiler via:

```
-DIREE_COMPILER_PLUGIN_PATHS=samples/compiler_plugins/simple_io_sample
```

It can then be activated in either `iree-opt` or `iree-compile` via the
option `--iree-plugin=simple_io_sample`.
