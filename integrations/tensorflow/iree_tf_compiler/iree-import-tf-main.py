import argparse
import re

import iree.compiler.tf
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


if __name__ == "__main__":
  Path(args.output_path).write_text(
    iree.compiler.tf.get_mlir(args.saved_model_path))
