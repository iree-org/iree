# TFLite integration tests status

This dashboard shows the models that are currently being tested on IREE's
presubmits.  If any tests are added or changed, please run
update_tflite_model_documentation.py to update this table.

|       Model        |      Status        |
| ------------------ | ------------------ |
cartoon_gan          | PASS ✓
east_text_detector   | PASS ✓
gpt2                 | PASS ✓
llvmcpu_mobilebert_tf2_quant | PASS ✓
llvmcpu_mobilenet_v1 | PASS ✓
llvmcpu_mobilenet_v3-large_uint8 | FAIL ✗
llvmcpu_posenet_i8   | FAIL ✗
llvmcpu_resnet_50_int8 | PASS ✓
mnasnet              | PASS ✓
mobilenet_v3         | PASS ✓
person_detect        | PASS ✓
vmvx_mobilebert_tf2_quant | PASS ✓
vmvx_mobilenet_v3-large_uint8 | FAIL ✗
vmvx_person_detect   | PASS ✓
vulkan_mobilebert_tf2_quant | FAIL ✗
vulkan_mobilenet_v1  | PASS ✓
vulkan_mobilenet_v3-large_uint8 | FAIL ✗
vulkan_posenet_i8    | FAIL ✗
vulkan_resnet_50_int8 | FAIL ✗