The IREE compiler transforms a model into its final deployable format in several
sequential steps. A model authored with Python in an ML framework should use the
corresponding framework's import tool to convert into a format (i.e.,
[MLIR](https://mlir.llvm.org/)) expected by the IREE compiler first.

Using a
[MobileNet model](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
as an example, import using IREE's [ONNX importer](../ml-frameworks/onnx.md):

```bash
# Download the model you want to compile and run.
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx

# Import to MLIR using IREE's ONNX importer.
pip install iree-base-compiler[onnx]
iree-import-onnx mobilenetv2-10.onnx --opset-version 17 -o mobilenetv2.mlir
```
