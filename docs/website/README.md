# IREE User-Facing Documentation Website

This directory contains the source and assets for IREE's website, hosted on
[GitHub Pages](https://pages.github.com/).

The website is generated using [MkDocs](https://www.mkdocs.org/), with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## How to edit this documentation

Follow <https://squidfunk.github.io/mkdocs-material/getting-started/> and read
<https://www.mkdocs.org/>.

Develop (from this folder):

```shell
mkdocs serve
```

Deploy:

* This force pushes to `gh-pages` on `<your remote>`. Please don't push to the
  main repository :)

```shell
mkdocs gh-deploy --remote-name <your remote>
```
