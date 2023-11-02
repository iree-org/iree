# IREE User-Facing Documentation Website

This directory contains the source and assets for IREE's website, hosted on
[GitHub Pages](https://pages.github.com/).

The website is generated using [MkDocs](https://www.mkdocs.org/), with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## How to edit this documentation

Follow <https://squidfunk.github.io/mkdocs-material/getting-started/> and read
<https://www.mkdocs.org/>.

All steps below are from this docs/website/ folder.

Setup (as needed):

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Develop:

```shell
./generate_extra_files.sh
mkdocs serve
```

Deploy:

* This force pushes to `gh-pages` on `<your remote>`. Please don't push to the
  main repository :)
* The `publish_website.yml` workflow takes care of publishing from the central
  repository automatically

```shell
mkdocs gh-deploy --remote-name <your remote>
```

## Website sections and authoring tips

For more details on how this is set up, see
[IREE Website Overview - July 10, 2023](https://docs.google.com/presentation/d/116TyW_aCsPXmmjRYI2tRqpOwDaGNoV8LDC_j9hsMrDk/edit?usp=sharing)
(though note that the website organization has changed since then).

For documentation language and style, the guide at
<https://developers.google.com/style> offers good advice.

### Building from source

Instructions on how to build the project from source on supported platforms.

* Focus on instructions that apply to all users, independent of specific
  package managers and development styles
* Set developers up for success with good default options
* Explain how interact with the build system and its outputs

### Guides

Workflow-oriented guides showing users how to accomplish tasks

### Reference

Unopinionated descriptions of system components

### Developers

Less structured pages for project development topics

* Pages may be "promoted" from this category to another category if they are
  generally useful to a wide enough range of developers

### Community (Blog)

A place to showcase work across the community
