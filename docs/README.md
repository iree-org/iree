# IREE Documentation

Documentation exclusively for project developers lives in
[`developers/`](developers/), while the source pages and assets for IREE's
user-focused website live in [`website/`](website/).

Developer documentation should use GitHub-flavored markdown (see
[this guide](https://guides.github.com/features/mastering-markdown/)), while
the website uses [MkDocs](https://www.mkdocs.org/), with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

A high bar should be set for pages published to the website:

* Page structure should be consistent across sections
* Content should be kept up to date regularly
* Instructions should be portable across environments (e.g. don't
  overspecialize on a specific Linux distribution or a particular version of
  Visual Studio on Windows)

When in doubt, the guide at https://developers.google.com/style offers good
instructions.

Developer documentation _can_ compromise on each of these points. Pages may
also be promoted to website/ after some refinement.
