The third_party/torch-mlir-dialects directory is used to import torch-mlir
specific dialects. This is needed transitively to provide dialects interfaces
to bridge between torch-mlir and IREE. After the related dialects are upstreamed
to mlir core as planned, this directory should no longer be needed.

## Upstream project link
https://github.com/llvm/torch-mlir

## Update command:
```shell
rsync -av --exclude=tools \
  --exclude=test \
  --exclude=lib/Dialect/TMTensor/Transforms \
  --exclude=include/torch-mlir-dialects/Dialect/TMTensor/Transforms \
  --exclude=README.md \
  PATH_TO_TORCH_MLIR_REPO/external/llvm-external-projects/torch-mlir-dialects \
  PATH_TO_IREE_REPO/third_party
```
