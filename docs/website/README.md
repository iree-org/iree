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
