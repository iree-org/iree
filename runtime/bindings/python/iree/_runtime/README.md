# Non user-API runtime support.

This package contains elements of the runtime which are not user
visible parts of the API. It is a pure source package that does
no native initialization by default (unlike the `runtime` package
which initializes the native library as part of exporting its API).

This distinction is important because some runtime libraries (like
`tracy` will attempt to connect to a profiler on load and there are
cases where we want to do things without triggering such load-time
behavior).

As peers to this package, there are native `_runtime_libs*` packages
for different variations of native libraries and tools.
