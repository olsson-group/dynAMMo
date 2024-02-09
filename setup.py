from setuptools import setup

setup(
    name="dynAMMo",
    version="0.0",
    packages=[
        "dynAMMo",
        "dynAMMo.base",
        "dynAMMo.model",
        "dynAMMo.tools",
        "dynAMMo.manifold",
    ],
    install_requires=[
        "deeptime==0.4.4",
        "h5py==3.9.0",
        "matplotlib==3.7.2",
        "MDAnalysis==2.7.0",
        "mdtraj==1.9.9",
        "msmtools==1.2.6",
        "networkx==3.2.1",
        "numpy==1.25.2",
        "pandas==2.0.3",
        "pyemma==2.5.12",
        "scipy==1.12.0",
        "setuptools==68.1.2",
        "torch==2.1.0",
        "tqdm==4.66.1",
    ],
    url="",
    license="MIT",
    author="Christopher Kolloff",
    author_email="kolloff@chalmers.se",
    description="dynAMMo – dynamic Augmented Markov Models",
)
