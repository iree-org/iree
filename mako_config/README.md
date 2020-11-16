This directory collects all Mako related files. The `uploader` is located at
[iree-mako](https://github.com/hanhanW/iree-mako/tree/iree). The Mako repo is
not pull in IREE because setting deps is complicated, and IREE only needs simple
features to upload performance data to Mako server.

To create a new benchmark, see
[Mako guide](https://github.com/google/mako/blob/master/docs/GUIDE.md) for more
details.
