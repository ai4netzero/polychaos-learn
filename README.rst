polychaos-learn
=====================

A simple Polynomial Chaos Expansions library built on top of `scikit-learn <https://scikit-learn.org>`_. The aim is provide a consistent interface with `sklearn.preprocessing.PolynomialFeatures <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_ to compose spectral basis functions (e.g. Legendre, Hermite Probabilistic, Hermite Physicist, Chebyshev, Laguerre) for regression like models. Additionally, a new fitting method is implemented targeting high dimensional problems.


Credits
-------

This development of this project received partial funding from the H2020 project `Enabling
onshore CO2 storage in Europe <http://www.enos-project.eu/>`__

The main contributors of this project are:

* Ahmed H. Elsheikh <a.elsheikh@hw.ac.uk>
* Alexander Tarakanov <tarakanov517@gmail.com>

Installation
------------

Dependencies
~~~~~~~~~~~~

scikit-learn requires:

- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Scikit-learn (>=0.20)

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of ``numpy``, ``scipy`` and ``scikit-learn``, 
you can check the latest sources with the command ::
	
	git clone https://github.com/ahmed-h-elsheikh/polychaos-learn

Then proceed with installation of ``polychaos-learn`` using ::

	cd polychaos-learn
	pip install .

Uninstall using using the following command ::
	
	pip uninstall polychaos-learn

Examples
~~~~~~~~~~~~~~~~~
Check the simple examples in this `jupyter notebook <https://github.com/ahmed-h-elsheikh/polychaos-learn/blob/master/examples/examples_v1.ipynb>`_. Also, you could run it on `google colab <https://colab.research.google.com/github/ahmed-h-elsheikh/polychaos-learn/blob/master/examples/examples_v1.ipynb>`_.


Citation
~~~~~~~~

If you use ``polychaos-learn`` in a scientific publication, we would appreciate to the following paper: 

.. parsed-literal::
	**Regression-based sparse polynomial chaos for uncertainty quantification of subsurface flow models**,
	Tarakanov Alexander and Elsheikh Ahmed H, Journal of Computational Physics, Volume 399, 15 December 2019,
	`https://doi.org/10.1016/j.jcp.2019.108909 <https://doi.org/10.1016/j.jcp.2019.108909>`_.


Bibtex entry:

.. parsed-literal::
	@article{polychaos-learn,
	     title={Regression-based sparse polynomial chaos for uncertainty quantification of subsurface flow models},
	     author={Alexander Tarakanov and Ahmed H. Elsheikh},
	     journal={Journal of Computational Physics},
	     volume={399},
	     year={2019},
	     issn={0021-9991},
	     doi={https://doi.org/10.1016/j.jcp.2019.108909},
	     year={2019},
	    }




