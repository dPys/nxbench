# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "nxbench"
author = "The NetworkX Developers"

# The full version, including alpha/beta/rc tags
release = "1.0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "*.json*",
    "*.pdf",
    "__pycache__",
    "test_*",
    "tests/*",
    "tests/test_*.py",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "furo"
html_title = "NxBench"
html_baseurl = "https://networkx.org"
html_copy_source = False
html_favicon = "_static/favicon.ico"
html_logo = "_static/nxbench_logo.png"
html_theme_options = {
    # "gtag": "G-XXXXXXXXXX",
    "source_repository": "https://github.com/dpys/nxbench/",
    "source_branch": "main",
    "source_directory": "doc/",
}

html_static_path = ["_static"]
html_css_files = ["css/nxbench-custom.css"]
html_js_files = ["js/nxbench-custom.js"]

# -- Autodoc Configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "exclude-members": "__weakref__",
    "special-members": "__init__",
    "inherited-members": False,
    "show-inheritance": False,
}

# -- Napoleon Configuration --------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Other Configurations ----------------------------------------------------

todo_include_todos = True

# -- Extension-specific Configuration -----------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Enable Markdown-specific features (optional)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Configure MyST-Parser (optional)
myst_heading_anchors = 3

nitpicky = True
nitpick_ignore = [
    # Missing References for pydantic_core.core_schema.*
    ("py:class", "pydantic_core.core_schema.AnySchema"),
    ("py:class", "pydantic_core.core_schema.NoneSchema"),
    ("py:class", "pydantic_core.core_schema.BoolSchema"),
    ("py:class", "pydantic_core.core_schema.IntSchema"),
    ("py:class", "pydantic_core.core_schema.FloatSchema"),
    ("py:class", "pydantic_core.core_schema.DecimalSchema"),
    ("py:class", "pydantic_core.core_schema.StringSchema"),
    ("py:class", "pydantic_core.core_schema.BytesSchema"),
    ("py:class", "pydantic_core.core_schema.DateSchema"),
    ("py:class", "pydantic_core.core_schema.TimeSchema"),
    ("py:class", "pydantic_core.core_schema.DatetimeSchema"),
    ("py:class", "pydantic_core.core_schema.TimedeltaSchema"),
    ("py:class", "pydantic_core.core_schema.LiteralSchema"),
    ("py:class", "pydantic_core.core_schema.EnumSchema"),
    ("py:class", "pydantic_core.core_schema.IsInstanceSchema"),
    ("py:class", "pydantic_core.core_schema.IsSubclassSchema"),
    ("py:class", "pydantic_core.core_schema.CallableSchema"),
    ("py:class", "pydantic_core.core_schema.ListSchema"),
    ("py:class", "pydantic_core.core_schema.TupleSchema"),
    ("py:class", "pydantic_core.core_schema.SetSchema"),
    ("py:class", "pydantic_core.core_schema.FrozenSetSchema"),
    ("py:class", "pydantic_core.core_schema.GeneratorSchema"),
    ("py:class", "pydantic_core.core_schema.DictSchema"),
    ("py:class", "pydantic_core.core_schema.AfterValidatorFunctionSchema"),
    ("py:class", "pydantic_core.core_schema.BeforeValidatorFunctionSchema"),
    ("py:class", "pydantic_core.core_schema.WrapValidatorFunctionSchema"),
    ("py:class", "pydantic_core.core_schema.PlainValidatorFunctionSchema"),
    ("py:class", "pydantic_core.core_schema.WithDefaultSchema"),
    ("py:class", "pydantic_core.core_schema.NullableSchema"),
    ("py:class", "pydantic_core.core_schema.UnionSchema"),
    ("py:class", "pydantic_core.core_schema.TaggedUnionSchema"),
    ("py:class", "pydantic_core.core_schema.ChainSchema"),
    ("py:class", "pydantic_core.core_schema.LaxOrStrictSchema"),
    ("py:class", "pydantic_core.core_schema.JsonOrPythonSchema"),
    ("py:class", "pydantic_core.core_schema.TypedDictSchema"),
    ("py:class", "pydantic_core.core_schema.ModelFieldsSchema"),
    ("py:class", "pydantic_core.core_schema.ModelSchema"),
    ("py:class", "pydantic_core.core_schema.DataclassArgsSchema"),
    ("py:class", "pydantic_core.core_schema.DataclassSchema"),
    ("py:class", "pydantic_core.core_schema.ArgumentsSchema"),
    ("py:class", "pydantic_core.core_schema.CallSchema"),
    ("py:class", "pydantic_core.core_schema.CustomErrorSchema"),
    ("py:class", "pydantic_core.core_schema.JsonSchema"),
    ("py:class", "pydantic_core.core_schema.UrlSchema"),
    ("py:class", "pydantic_core.core_schema.MultiHostUrlSchema"),
    ("py:class", "pydantic_core.core_schema.DefinitionsSchema"),
    ("py:class", "pydantic_core.core_schema.DefinitionReferenceSchema"),
    ("py:class", "pydantic_core.core_schema.UuidSchema"),
    # Missing References for typing.Self
    ("py:class", "typing.Self"),
    # Missing References for nx.Graph
    ("py:class", "nx.Graph"),
    ("py:class", "nx.MultiDiGraph"),
    ("py:class", "nx.MultiGraph"),
    # Missing References for fsspec.spec.AbstractFileSystem
    ("py:class", "fsspec.spec.AbstractFileSystem"),
    # Missing References for optional
    ("py:class", "optional"),
    # Additional Missing References
    ("py:class", "np.ndarray"),
    # Additional missing references as per warnings
    ("py:class", "pydantic.deprecated.parse.Protocol"),
    ("py:class", "pattern='^"),
    ("py:class", "$'"),
    ("py:class", "types.Annotated"),
    ("py:class", "annotated_types.Gt"),
    ("py:class", "gt=0"),
    # Missing References
    ("py:class", "nxbench.config.nxbenchConfig"),
    ("py:class", "Path"),
    ("py:class", "Document"),
    ("py:class", "function"),
    ("py:class", "click.ClickException"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "bs4.BeautifulSoup"),
]

autodoc_inherit_docstrings = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "psutil": ("https://psutil.readthedocs.io/en/latest/", None),
}

autodoc_typehints = "description"
