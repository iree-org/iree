# TFLite integration tests status

This dashboard shows the models that are currently being tested on IREE's
presubmits.  If any tests are added or changed, please run
update_tflite_model_documentation.py to update this table.

|       Model        |      Status        |
| ------------------ | ------------------ |
vulkan_mobilenet_v1  | PASS ✓
llvmcpu_mobilenet_v3-large_uint8 | FAIL ✗
vmvx_mobilebert_tf2_quant | PASS ✓
llvmcpu_mobilebert_tf2_quant | PASS ✓
vulkan_mobilenet_v3-large_uint8 | FAIL ✗
person_detect        | PASS ✓
cartoon_gan          | PASS ✓
vulkan_resnet_50_int8 | FAIL ✗
east_text_detector   | FAIL ✗
vmvx_person_detect   | PASS ✓
gpt2                 | PASS ✓
mobilenet_v3         | PASS ✓
llvmcpu_resnet_50_int8 | PASS ✓
llvmcpu_posenet_i8   | FAIL ✗
mnasnet              | PASS ✓
vulkan_posenet_i8    | FAIL ✗
vmvx_mobilenet_v3-large_uint8 | FAIL ✗
vulkan_mobilebert_tf2_quant | FAIL ✗
llvmcpu_mobilenet_v1 | FAIL ✗