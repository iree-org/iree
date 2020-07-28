# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


def complex_add(a_re, a_im, b_re, b_im):
  return a_re + b_re, a_im + b_im


def complex_mul(a_re, a_im, b_re, b_im):
  c_re = a_re * b_re - a_im * b_im
  c_im = a_re * b_im + a_im * b_re
  return c_re, c_im


# This is a fun but quite interesting example because the return value and most
# of the interior computations are dynamically shaped.
class MandelbrotModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.int32),
      tf.TensorSpec([], tf.int32)
  ])
  def calculate(self, center_re, center_im, view_size, view_pixels,
                num_iterations):
    """Calculates an image which represents the Mandelbrot set.

    Args:
      center_re: The center point of the view (real part).
      center_im: The center point of the view (imaginary part).
      view_size: The view will display a square with this size.
      view_pixels: The returned image will be a square with this many pixels on
        a side.
      num_iterations: The number of iterations to use for determining escape.

    Returns:
      A tensor of pixels with shape [view_size, view_size] which represents
      the mandelbrot set.
    """
    re_min = center_re - view_size / 2.
    re_max = center_re + view_size / 2.
    im_min = center_im - view_size / 2.
    im_max = center_im + view_size / 2.
    re_coords = tf.linspace(re_min, re_max, view_pixels)
    im_coords = tf.linspace(im_min, im_max, view_pixels)

    # Generate flat list of real and imaginary parts of the points to test.
    # This requires taking all pairs of re_coords and im_coords, which we
    # do by broadcasting into a 2d matrix (real part is broadcasted "vertically"
    # and imaginary part is broadcasted "horizontally").
    # We use a Nx1 * 1xN -> NxN matmul to do the broadcast.
    c_re = tf.reshape(
        tf.matmul(
            tf.ones([view_pixels, 1]), tf.reshape(re_coords, [1, view_pixels])),
        [-1])
    c_im = tf.reshape(
        tf.matmul(
            tf.reshape(im_coords, [view_pixels, 1]), tf.ones([1, view_pixels])),
        [-1])

    z_re = tf.zeros_like(c_re)
    z_im = tf.zeros_like(c_im)
    for _ in range(num_iterations):
      square_re, square_im = complex_mul(z_re, z_im, z_re, z_im)
      z_re, z_im = complex_add(square_re, square_im, c_re, c_im)

    # Calculate if the points are in the set (that is, if their orbit under the
    # recurrence relationship has diverged).
    z_abs = tf.sqrt(z_re**2 + z_im**2)
    z_abs = tf.where(tf.math.is_nan(z_abs), 100. * tf.ones_like(z_abs), z_abs)
    in_the_set = tf.where(z_abs > 50., tf.ones_like(z_abs),
                          tf.zeros_like(z_abs))
    # Return an image
    return tf.reshape(in_the_set, shape=[view_pixels, view_pixels])


@tf_test_utils.compile_module(MandelbrotModule)
class MandelbrotTest(tf_test_utils.TracedModuleTestCase):

  def test_mandelbrot(self):

    def mandelbrot(module):
      # Basic view of the entire set.
      module.calculate(-0.7, 0.0, 3.0, 400, 100)
      # This is a much more detailed view, so more iterations are needed.
      module.calculate(-0.7436447860, 0.1318252536, 0.0000029336, 400, 3000)

    self.compare_backends(mandelbrot)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
