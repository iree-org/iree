# TFLite integration tests status

This dashboard shows the models that are currently being tested on IREE's
presubmits.  If any tests are added or changed, please run
update_tflite_model_documentation.py to update this table.

|       Model        |      Status        |
| ------------------ | ------------------ |
person_detect        | <span class="success-table-element">✓</span>
east_text_detector   | <span class="success-table-element">✓</span>
vulkan_posenet_i8    | <span class="failure-table-element">✗</span>
cartoon_gan          | <span class="success-table-element">✓</span>
mnasnet              | <span class="success-table-element">✓</span>
gpt2                 | <span class="success-table-element">✓</span>
llvmaot_posenet_i8   | <span class="success-table-element">✓</span>
mobilenet_v3         | <span class="success-table-element">✓</span>
llvmaot_mobilenet_v1 | <span class="success-table-element">✓</span>
vulkan_mobilenet_v1  | <span class="failure-table-element">✗</span>