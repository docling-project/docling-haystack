# Haystack Docling integration

[![PyPI version](https://img.shields.io/pypi/v/docling-haystack)](https://pypi.org/project/docling-haystack/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-haystack)](https://pypi.org/project/docling-haystack/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/DS4SD/docling)](https://opensource.org/licenses/MIT)

A [Docling](https://github.com/DS4SD/docling) integration for
[Haystack](https://github.com/deepset-ai/haystack/).

## Installation

Simply install `docling-haystack` from your package manager, e.g. pip:
```bash
pip install docling-haystack
```

## Usage

### Basic usage

Basic usage of `DoclingConverter` looks as follows:

```python
from haystack import Pipeline
from docling_haystack.converter import DoclingConverter

idx_pipe = Pipeline()
# ...
converter = DoclingConverter()
idx_pipe.add_component("converter", converter)
# ...
```
### Advanced usage

When initializing a `DoclingConverter`, you can use the following parameters:

- `converter` (optional): any specific Docling `DocumentConverter` instance to use
- `convert_kwargs` (optional): any specific kwargs for conversion execution
- `export_type` (optional): export mode to use: `ExportType.DOC_CHUNKS` (default) or
    `ExportType.MARKDOWN`
- `md_export_kwargs` (optional): any specific Markdown export kwargs (for Markdown mode)
- `chunker` (optional): any specific Docling chunker instance to use (for doc-chunk
    mode)
- `meta_extractor` (optional): any specific metadata extractor to use

### Example

For an end-to-end usage example, check out
[this notebook](https://docling-project.github.io/docling/examples/rag_haystack/).
