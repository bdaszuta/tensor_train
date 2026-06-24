# Sphinx configuration for tensor_train
# Breathe bridges Doxygen XML output into Sphinx RST.

project = 'tensor_train'
copyright = '2026, Boris Daszuta'
author = 'Boris Daszuta'
release = '0.1.0'

extensions = [
    'breathe',
    'sphinx.ext.mathjax',
]

breathe_projects = {
    'tensor_train': '_doxygen/xml',
}
breathe_default_project = 'tensor_train'

primary_domain = 'cpp'
highlight_language = 'cpp'

templates_path = ['_templates']
exclude_patterns = ['_build', '_doxygen', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

breathe_show_define_initializer = True
breathe_implementation_filename_extensions = ['.hpp']

# doxygennamespace pulls in classes also shown on dedicated pages.
# The Python bindings page has overloaded functions which Sphinx flags
# as duplicate object descriptions.
suppress_warnings = ["duplicate_declaration.cpp",
                     "duplicate_object_description",
                     "duplicate_label"]
