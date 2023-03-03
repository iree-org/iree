import re

import tensorflow as tf
from tensorflow.python import pywrap_mlir
from pathlib import Path

def convert_to_hlo(model_path: str, use_stablehlo=True):
  result = pywrap_mlir.experimental_convert_saved_model_to_mlir(
      model_path, "", show_debug_info=False)

  # See:
  # * https://github.com/tensorflow/tensorflow/issues/59685
  # * https://github.com/tensorflow/tensorflow/blob/cd5667ec9d8af787a18c5b1ae239f060e9fa6fdd/tensorflow/python/saved_model/function_deserialization.py#L641-L653
  result = re.sub(r"__inference_(.*)_\d+", r"\1", result)

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