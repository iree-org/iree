# IREE Common Transformations

This directory contains common transformations and passes that are useful across
different phases of the IREE compiler (e.g., Global Optimization, HAL
Transformation, Codegen).

Passes in this directory should generally be self-contained and avoid
dependencies on specific backend dialects (like the Codegen dialect) where
possible. They often serve as wrappers around upstream MLIR passes or implement
IREE-specific but backend-agnostic transformations.
