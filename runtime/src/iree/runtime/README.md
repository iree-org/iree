# IREE Higher-Level Runtime API

This directory implements a higher-level runtime API on top of the low level
APIs split across `iree/base/api.h`, `iree/hal/api.h`, and `iree/vm/api.h`.

Using this higher level API may pull in additional dependencies and perform
additional allocations compared to what you can get by directly going to the
lower levels. For the most part, the higher level and lower levels APIs may be
mixed.

See [the demo directory](./demo/) for sample usage.
