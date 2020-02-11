# VMLA (Virtual Machine-based Linear Algebra)

This dialect is designed to closely model XLA HLO ops in a way that is easy to
map to execution on the IREE VM. The changes involve using byte buffers instead
of tensors, propagating shape information and converting shape math to simple
integer arithmetic, and legalizing types to supported values (such as 1bit bools
to 8bit integers of 0 or 1).

## Adding an Op

TODO(benvanik): document and show an example change.
