import setuptools
from os import path
import pathlib
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Get the long description from the README file
with open(path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

requirementPath = path.join(HERE, '/requirements.txt')
install_requires = ["gdal==2.3.3", "netcdf4==1.4.2", "h5py==2.9.0", "numpy==1.15.4", "OSR==0.0.1", "scipy==1.1.0", "shapely==1.6.4"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="thyme",
    version="0.1.0",
    license="",
    author="Erin Nagel, Jason Greenlaw",
    author_email="erin.nagel@noaa.gov, jason.greenlaw@noaa.gov",
    description="Tools for Hydrodynamic Model Output Extraction and Processing",
    long_description=long_description,
    url="",
    packages=setuptools.find_packages(),
    install_requires=['gdal', 'numpy', 'shapely', 'osr', 'h5py','netCDF4', 'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
)
