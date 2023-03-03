import argparse
import re

import tensorflow as tf
from tensorflow.python import pywrap_mlir
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p',
                    '--saved_model_path',
                    dest='saved_model_path',
                    required=True,
                    help='Path to the saved model directory to import.')
parser.add_argument('-o',
                    '--output_path',
                    dest='output_path',
                    required=True,
                    help='Path to the mlir file name to output.')
args = parser.parse_args()


def convert_to_hlo(model_path: str):
  result = pywrap_mlir.experimental_convert_saved_model_to_mlir(
      model_path, "", show_debug_info=False)

  # See:
  # * https://github.com/tensorflow/tensorflow/issues/59685
  # * https://github.com/tensorflow/tensorflow/blob/cd5667ec9d8af787a18c5b1ae239f060e9fa6fdd/tensorflow/python/saved_model/function_deserialization.py#L641-L653
  result = re.sub(r"__inference_(.*)_\d+", r"\1", result)

  pipeline = ["tf-lower-to-mlprogram-and-hlo"]
  result = pywrap_mlir.experimental_run_pass_pipeline(result,
                                                      ",".join(pipeline),
                                                      show_debug_info=False)
  return result


if __name__ == "__main__":
  Path(args.output_path).write_text(convert_to_hlo(args.saved_model_path))
