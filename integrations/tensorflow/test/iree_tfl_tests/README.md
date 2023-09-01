# TFLite integration tests status

This dashboard shows the models that are currently being tested on IREE's
presubmits.  If any tests are added or changed, please run
update_tflite_model_documentation.py to update this table.

|       Model        |      Status        |
| ------------------ | ------------------ |
mobilenet_v3         | PASS ✓
llvmcpu_resnet_50_int8 | PASS ✓
vulkan_mobilebert_tf2_quant | FAIL ✗
cartoon_gan          | PASS ✓
llvmcpu_mobilebert_tf2_quant | PASS ✓
mnasnet              | PASS ✓
person_detect        | PASS ✓
vulkan_posenet_i8    | FAIL ✗
east_text_detector   | PASS ✓
gpt2                 | PASS ✓
llvmcpu_mobilenet_v1 | PASS ✓
llvmcpu_mobilenet_v3-large_uint8 | PASS ✓
vulkan_mobilenet_v1  | PASS ✓
vulkan_mobilenet_v3-large_uint8 | FAIL ✗
llvmcpu_posenet_i8   | FAIL ✗
vulkan_resnet_50_int8 | FAIL ✗