# Copyright 2020 Google LLC
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

# Special cases to exclude from automatically expanding targets for all
# backends.
set(SPECIAL_CASES
    "mobile_bert_squad_test.py"
)

set(TFLITE_FAILING
    "broadcasting_test.py"
    "concat_test.py"
    "dynamic_mlp_relu_test.py"
    "dynamic_mlp_test.py"
    "einsum_dynamic_test.py"
    "einsum_static_test.py"
    "einsum_vector_test.py"
    "fft_test.py"
    "gather_test.py"
    "image_resize_test.py"
    "mandelbrot_test.py"
    "matrix_ops_dynamic_test.py"
    "quantization_dyn_test.py"
    "resource_ops_test.py"
    "ring_buffer_test.py"
    "scatter_update_test.py"
    "simple_stateful_test.py"
    "sliding_window_test.py"
    "strings_test.py"
)

set(VMLA_FAILING
    "einsum_dynamic_test.py"
    "einsum_static_test.py"
    "einsum_vector_test.py"
    "mandelbrot_test.py"  # TODO(silvasean): Get this working on IREE.
    "ring_buffer_test.py"  # TODO(b/148747011)
    "strings_test.py"
    "tensorlist_test.py"  # TODO(suderman): Re-enable once dependencies resolved
)

set(LLVM_FAILING
    "broadcast_to_test.py"
    "broadcasting_test.py"
    "conv_transpose_test.py"
    "dynamic_mlp_relu_test.py"
    "dynamic_mlp_test.py"
    "einsum_dynamic_test.py"
    "einsum_static_test.py"
    "einsum_vector_test.py"
    "fft_test.py"  # TODO(natashaknk): Get this working after kernel is in.
    "fill_test.py"  # TODO(jennik): Get this test working on IREE.
    "linspace_test.py"  # TODO(https://github.com/google/iree/issues/1521)
    "mandelbrot_test.py"  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py"
    "quantization_dyn_test.py"
    "range_test.py"
    "ring_buffer_test.py"  # TODO(b/148747011)
    "scatter_update_test.py"
    "strings_test.py"
    "tensorlist_test.py"  # TODO(suderman): Re-enable once dependencies resolved
)

set(VULKAN_FAILING
    "broadcast_to_test.py"
    "broadcasting_test.py"
    "conv_transpose_test.py"
    "dynamic_mlp_relu_test.py"
    "dynamic_mlp_test.py"
    "einsum_dynamic_test.py"
    "einsum_static_test.py"
    "einsum_vector_test.py"
    "fft_test.py"  # TODO(natashaknk): Get this working after kernel is in.
    "fill_test.py"  # TODO(jennik): Get this test working on IREE.
    "linspace_test.py"  # TODO(https://github.com/google/iree/issues/1521)
    "mandelbrot_test.py"  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py"
    "quantization_dyn_test.py"
    "range_test.py"
    "ring_buffer_test.py"  # TODO(b/148747011)
    "scatter_update_test.py"
    "strings_test.py"
    "tensorlist_test.py"  # TODO(suderman): Re-enable once dependencies resolved
)

set(REFERENCE_BACKEND tf)

function(add_e2e_test_backend filename target_backend labels)
  set(_name "tensorflow_e2e__${filename}__${target_backend}")
  add_test(
    NAME
      ${_name}
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND
      "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
      "--reference_backend=${REFERENCE_BACKEND}"
      "--target_backends=${target_backend}"
  )
  set_property(TEST ${_name} PROPERTY LABELS "${labels}")
  set_property(TEST ${_name} PROPERTY ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")
endfunction()

function(add_e2e_test filename)
  # Exclude special files.
  if("${filename}" IN_LIST SPECIAL_CASES)
    return()
  endif()

  # Build target_backends
  if(NOT "${filename}" IN_LIST VMLA_FAILING)
    add_e2e_test_backend("${filename}" iree_vmla "")
  endif()
  if(NOT "${filename}" IN_LIST LLVM_FAILING)
    add_e2e_test_backend("${filename}" iree_llvmaot "driver=dylib")
  endif()
  if(NOT "${filename}" IN_LIST VULKAN_FAILING)
    add_e2e_test_backend("${filename}" iree_vulkan "driver=vulkan")
  endif()
  if(NOT "${filename}" IN_LIST TFLITE_FAILING)
    add_e2e_test_backend("${filename}" tflite "")
  endif()
endfunction()

function(add_all_e2e_tests pattern)
  file(GLOB _all_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${pattern}")
  foreach(_filename ${_all_files})
    add_e2e_test("${_filename}")
  endforeach()
endfunction()

add_all_e2e_tests("*_test.py")
