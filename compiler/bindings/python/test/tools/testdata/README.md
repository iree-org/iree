# Importer test data.

Most files have a generation script except for when it is expected that they
will never change. Things in that category and break glass instructions to
update:

* LeakyReLU.onnx: Just a random single-op ONNX test to verify that the upstream
  importer is wired properly. It should never need to be updated but if it
  does, pretty much any single-op test case from the ONNX test suite will
  suffice.
