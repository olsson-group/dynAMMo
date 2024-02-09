# dynAMMo: Dynamic Augmented Markov Models

dynAMMo is a Python library designed to enhance the capabilities of traditional Markov Models by incorporating dynamic observables and providing tools for the analysis of biomolecular dynamics through augmented Markov models.

## Installation

dynAMMo can be installed directly from PyPI using pip:

```bash
pip install dynammo
```

Ensure you have Python 3.9 or newer installed. For a development version or to contribute, clone the repository and install the package using:

```bash
git clone https://github.com/olsson-group/dynAMMo.git
cd dynammo
pip install .
```

## Quick Start

Check out the tutorial (`tutorial.ipynb`) for usage.

## Documentation

In progress

## Contributing

We welcome contributions to dynAMMo! If you're interested in contributing, please follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or fix.
4. Make your changes.
5. Commit your changes and push them to your fork.
6. Submit a pull request from your fork back to the main dynAMMo repository.

## License

dynAMMo is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.

## Citation

If you use dynAMMo in your research, please cite it as follows:

```bibtex
@article{Kolloff2023,
doi = {10.1088/2632-2153/ad10ce},
url = {https://dx.doi.org/10.1088/2632-2153/ad10ce},
year = {2023},
month = {dec},
publisher = {IOP Publishing},
volume = {4},
number = {4},
pages = {045050},
author = {Christopher Kolloff and Simon Olsson},
title = {Rescuing off-equilibrium simulation data through dynamic experimental data with dynAMMo},
journal = {Machine Learning: Science and Technology},
abstract = {Long-timescale behavior of proteins is fundamental to many biological processes. Molecular dynamics (MD) simulations and biophysical experiments are often used to study protein dynamics. However, high computational demands of MD limit what timescales are feasible to study, often missing rare events, which are critical to explain experiments. On the other hand, experiments are limited by low resolution. We present dynamic augmented Markov models (dynAMMo) to bridge the gap between these data and overcome their respective limitations. For the first time, dynAMMo enables the construction of mechanistic models of slow exchange processes that have been not observed in MD data by integrating dynamic experimental observables. As a consequence, dynAMMo allows us to bypass costly and extensive simulations, yet providing mechanistic insights of the system. Validated with controlled model systems and a well-studied protein, dynAMMo offers a new approach to quantitatively model protein dynamics on long timescales in an unprecedented manner.}
}

```

## Contact

For questions or support, please open an issue on the GitHub repository, or contact us directly at kolloff@chalmers.se.
```