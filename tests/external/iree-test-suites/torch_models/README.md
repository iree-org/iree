# Torch Models Test Suite

## Directory Structure

```
torch_models/
├── model1/
│   ├── modules/
│   │   ├── module1.json
│   │   ├── module2.json
│   │   └── ...
│   ├── test1.json
│   ├── test2.json
│   └── ...
├── model2/
│   ├── modules/
│   │   ├── module1.json
│   │   ├── module2.json
│   │   └── ...
│   ├── test1.json
│   ├── test2.json
│   └── ...
```

## Markers

We try to add markers to every test. The CI collects tests for a machine based
on these markers.

- Add a marker for the model class. For example "sdxl".
- Add a marker for the compilation backend for the test. For example
  "llvm-cpu", "hip", "spirv".
- Add a compilation target specific marker if the test is sku independent. For
  example: "gfx942", "gfx1201", "sm80". This is generally try for quality and
  compstat tests, unless using sku specific tuning specs.
- Add a sku specific marker if the test is sku specific. For example: "mi325",
  "w7900", "rtx4090". This is generally true for all benchmarking tests and
  when using sku specific tuner files.

## Misc

Information on accepted test schema: https://github.com/iree-org/iree-test-suites/blob/main/torch_models/README.md
