# Tensor Operator Set Architecture

[TOSA](https://developer.mlplatform.org/w/tosa/) defines a set of common
tensor operations to most machine learning
frameworks. TOSA's design has been focused provide a simple intermediate
representation for ingesting models from a variety of sources while
guaranteeing the representation can support generating efficient
execution on CPUs, GPUs, and custom accelerators. These operations
support both floating point and fixed-point (quantized) operations.

IREE uses TOSA as a prioritized ingestion dialect, transforming multiple
ML-platform ingestion formats into a TOSA compatible set of operations.
To propose enhancements / changes to the TOSA specification submit a
proposal on TOSA's platform development
[page](https://developer.mlplatform.org/w/tosa/#:~:text=Specification%20Contributions)
