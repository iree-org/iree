import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/default/1?lite-format=tflite"

class MobilenetV1Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobilenetV1Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(MobilenetV1Test, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=1e-4).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

