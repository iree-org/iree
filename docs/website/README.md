# IREE User-Facing Documentation Website

This directory contains the source and assets for <https://iree.dev/>.

The website is generated using [MkDocs](https://www.mkdocs.org/), with the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme and
is served using [GitHub Pages](https://pages.github.com/).

## How to edit this documentation

Follow <https://squidfunk.github.io/mkdocs-material/getting-started/> and read
<https://www.mkdocs.org/>.

All steps below are from this `docs/website/` folder.

**Setup** as needed:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Develop** by building and running a local webserver:

```shell
# Optionally generate extra files, e.g. MLIR dialect pages produced by tablegen
# The website will still work without these pages.
# ./generate_extra_files.sh

mkdocs serve
```

### Deploying the documentation

The
[`publish_website.yml`](https://github.com/iree-org/iree/actions/workflows/publish_website.yml)
workflow in <https://github.com/iree-org/iree> builds the website and deploys
it automatically.

If you want to host the documentation on your own GitHub Pages site, such as to
allow code reviewers to preview your changes, you can deploy the documentation
yourself.

> [!IMPORTANT]
> If you are just previewing changes locally, you don't need this!
>
> This force pushes to the `gh-pages` branch on `<your remote>`. Please don't
> push to the main <https://github.com/iree-org/iree> repository :)

To deploy to a remote:

```shell
mkdocs gh-deploy --remote-name <your remote>
```

Assuming you have GitHub Pages configured for your remote from the `gh-pages`
branch (see `https://github.com/<your remote>/iree/settings/pages`), the
website should be live shortly at `https://<your remote>.github.io/iree`.

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
