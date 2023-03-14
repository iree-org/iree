# RUN: %PYTHON %s

import absl.testing
import numpy
from . import imagenet_test_data
from . import test_util

# Model is INT8 quantized but inputs and outputs are FP32.
model_path = "https://storage.googleapis.com/tf_model_garden/vision/resnet50_imagenet/resnet_50_224_int8.tflite"


class Resnet50Int8Test(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(Resnet50Int8Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(Resnet50Int8Test, self).compare_results(iree_results, tflite_results,
                                                  details)
    self.assertTrue(numpy.isclose(iree_results, tflite_results, atol=0.3).all())

  def generate_inputs(self, input_details):
    inputs = imagenet_test_data.generate_input(self.workdir, input_details)
    # Normalize inputs to [-1, 1].
    inputs = (inputs.astype('float32') / 127.5) - 1
    return [inputs]

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
