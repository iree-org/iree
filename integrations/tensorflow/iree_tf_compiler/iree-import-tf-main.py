import tensorflow as tf
from tensorflow.python import pywrap_mlir
from pathlib import Path

def convert_to_hlo(model_path: str, use_stablehlo=True):
  result = pywrap_mlir.experimental_convert_saved_model_to_mlir(
      model_path, "", show_debug_info=False)
  pipeline = ["tf-lower-to-mlprogram-and-hlo"]
  if not use_stablehlo:
    pipeline.append("stablehlo-legalize-to-hlo")
  result = pywrap_mlir.experimental_run_pass_pipeline(
      result, ",".join(pipeline), show_debug_info=False)
  return result

Path("/tmp/simple-model-new.stablehlo.mlir").write_text(
  convert_to_hlo("/tmp/simple-model", True))
Path("/tmp/simple-model-new.hlo.mlir").write_text(
  convert_to_hlo("/tmp/simple-model", False))