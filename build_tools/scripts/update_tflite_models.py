# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This tool handles mirroring tflite testing files from their source to the
# the iree-model-artifacts test bucket. This avoids taking dependency on
# external test files that may change or no longer be available.
#
# To update all files:
#   python update_tflite_models.py --file all
#
# To update a specific file:
#   python update_tflite_models.py --file posenet_i8_input.jpg
#
# Note you must have write permission to the iree-model-artifacts GCS bucket
# with local gcloud authentication.

from absl import app
from absl import flags
from google.cloud import storage
from google_auth_oauthlib import flow

import tempfile
import urllib

FLAGS = flags.FLAGS
flags.DEFINE_string('file', '', 'file to update')

file_dict = dict({
    "mobilenet_v1.tflite":
        "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/default/1?lite-format=tflite",
    "posenet_i8.tflite":
        "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
    "posenet_i8_input.jpg":
        "https://github.com/tensorflow/examples/raw/master/lite/examples/pose_estimation/raspberry_pi/test_data/image3.jpeg"
})

BUCKET_NAME = "iree-model-artifacts"
FOLDER_NAME = "tflite-integration-tests"


def upload_model(source, destination, tmpfile):
  """Uploads a file to the bucket."""
  urllib.request.urlretrieve(source, tmpfile)

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(BUCKET_NAME)
  blob = bucket.blob("/".join([FOLDER_NAME, destination]))
  blob.upload_from_filename(tmpfile)


def main(argv):
  tf = tempfile.NamedTemporaryFile()

  items = file_dict.items()

  if FLAGS.file in file_dict:
    items = [(FLAGS.file, file_dict[FLAGS.file])]
  elif FLAGS.file != "all":
    print('Unknown file to upload: ', "\"" + FLAGS.file + "\"")
    exit()

  for dst, src in items:
    upload_model(src, dst, tf.name)


if __name__ == '__main__':
  app.run(main)
