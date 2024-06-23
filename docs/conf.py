# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('/home/yutinlin/workspace/iMSminer/src/'))
scripts = os.listdir(r"/home/yutinlin/workspace/iMSminer/src/iMSminer/")
scripts = [item for item in scripts if ".py" in item]
for script in scripts:
    if f"/home/yutinlin/workspace/iMSminer/src/iMSminer/{script}" not in sys.path:
        sys.path.append(
            f"/home/yutinlin/workspace/iMSminer/src/iMSminer/{script}")
imsminer_site_packages = os.listdir(
    r"/home/yutinlin/anaconda3/lib/python3.11/site-packages/iMSminer")
if f"/home/yutinlin/anaconda3/lib/python3.11/site-packages/iMSminer/{imsminer_site_packages}" not in sys.path:
    sys.path.append(
        f"/home/yutinlin/anaconda3/lib/python3.11/site-packages/iMSminer/{imsminer_site_packages}")
imsminer_site_packages = os.listdir(
    r"/home/yutinlin/anaconda3/lib/python3.11/site-packages")
if f"/home/yutinlin/anaconda3/lib/python3.11/site-packages/{imsminer_site_packages}" not in sys.path:
    sys.path.append(
        f"/home/yutinlin/anaconda3/lib/python3.11/site-packages/{imsminer_site_packages}")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'iMSminer'
copyright = '2024, Yu Tin Lin'
author = 'Yu Tin Lin'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'renku'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
