import setuptools
import pathlib
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Get the long description from the README file
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='thyme',
    license='BSD-2-Clause',
    version='0.4.5',
    author='Erin Nagel, Jason Greenlaw',
    author_email='erin.nagel@noaa.gov, jason.greenlaw@noaa.gov',
    description='Tools for Hydrodynamic Model Output Extraction and Processing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/noaa-ocs-modeling/thyme',
    setup_requires=['numpy'],
    install_requires=['shapely', 'numpy', 'scipy', 'netCDF4', 'gdal'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
