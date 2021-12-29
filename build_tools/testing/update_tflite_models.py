from absl import app
from absl import flags
from google.cloud import storage
from google_auth_oauthlib import flow

import tempfile
import urllib

FLAGS = flags.FLAGS

flatbuffers = dict({
    "mobilenet_v1.tflite":
        "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/default/1?lite-format=tflite",
    "posenet_i8.tflite":
        "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
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

  for dst, src in flatbuffers.items():
    upload_model(src, dst, tf.name)


if __name__ == '__main__':
  app.run(main)
