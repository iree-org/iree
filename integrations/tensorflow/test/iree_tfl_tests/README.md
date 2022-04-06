# TFLite integration tests status

This dashboard shows the models that are currently being tested on IREE's
presubmits.  If any tests are added or changed, please run
update_tflite_model_documentation.py to update this table.

|       Model        |      Status        |
| ------------------ | ------------------ |
person_detect        | PASS ✓
east_text_detector   | PASS ✓
vulkan_posenet_i8    | FAIL ✗
cartoon_gan          | PASS ✓
mnasnet              | PASS ✓
gpt2                 | PASS ✓
llvmaot_posenet_i8   | PASS ✓
mobilenet_v3         | PASS ✓
llvmaot_mobilenet_v1 | PASS ✓
vulkan_mobilenet_v1  | FAIL ✗