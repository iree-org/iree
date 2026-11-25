# IREE PyTorch Importer

`iree.importers.torch` imports PyTorch programs into IREE. It takes programs in
the `torch.export` form (`ExportedProgram`), either exported in the same process
or loaded from a `.pt2` file, and produces a module that `iree-compile` accepts.
It is distributed as the `iree-importer-torch` wheel.

`iree` is a namespace package shared with `iree.compiler`, `iree.runtime` and
`iree.core`, so there must be no `iree/__init__.py` and no
`iree/importers/__init__.py`.

## Importing a saved program

```shell
iree-import-torch model.pt2 -o model.mlir
iree-compile model.mlir --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu -o model.vmfb
```

Several programs become the functions of one module. Tensors that the programs
hold under the same name with the same contents are imported once, so that
programs saved separately share their weights and state:

```shell
iree-import-torch prefill=prefill.pt2 decode=decode.pt2 -o model.mlirbc \
    --externalize-params --save-params-to model.irpa
```

`--externalize-params` keeps the weights out of the module and references them
by name; `--save-params-to` writes them to a parameter archive, which needs
`iree.runtime`.

A program that uses custom operators or pytree types needs them registered
before it is loaded. Name the module that registers them with `--import-module`.
IREE cannot import a custom operator that it has no lowering for. The module
can make the importer decompose it, or any other operator, with
`register_import_decomposition`.

The same is available from Python:

```python
from iree.importers.torch import import_programs, load_pt2

output = import_programs(
    {"main": load_pt2("model.pt2")},
    externalize_parameters=True,
    save_parameters_to="model.irpa",
)
output.lower_to_iree()
output.save_mlir("model.mlirbc")
```

## Testing

The tests need `iree.compiler` and `iree.runtime`, from a build or from the
wheels, and the packages in `test/requirements.txt`:

```shell
python -m pip install -r integrations/torch/python/test/requirements.txt
python -m pytest integrations/torch/python/test
```
