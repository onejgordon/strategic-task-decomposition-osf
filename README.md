# Materials for "Strategic task decomposition in joint action" (Cognitive Science, 2023)

## Manifest

* Pandas data frames are in the `data/` directory.
* Notebooks for reproduction of figures is labeled by figure number and found in the `notebooks/` directory.
* SPSS Syntax files for ANOVA analyses are in the `spss_scripts/` directory.

## Common Setup Issues

1. Failed installing scipy/numpy

Install using conda from nightly: `pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy`

2. Similar problem with shapely

`brew install geos`, followed by `pip install shapely --no-cache-dir`?

