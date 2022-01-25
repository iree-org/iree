# IREE Hardware Abstraction Layer (HAL)

The IREE HAL expresses a low-level abstraction over modern compute APIs like
Vulkan (CPUs count too!). Each implementation of the HAL interface can:

* Enumerate and query devices and their capabilities
* Define executable code that runs on the device
* Allocate unified or discrete memory and provide cache control
* Organize work into sequences for deferred submission
* Provide explicit synchronization primitives for ordering submissions

Refer to IREE's
[presentations and talks](../../README.md#presentations-and-talks) for further
details.

## Testing

See the [cts/ folder](./cts/) for the HAL Conformance Test Suite.
