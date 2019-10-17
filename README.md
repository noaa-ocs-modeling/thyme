thyme
=====
[![Build Status](https://travis-ci.com/noaa-ocs-modeling/thyme.svg?branch=master)](https://travis-ci.com/noaa-ocs-modeling/thyme)

**T**ools for **Hy**drodynamic **M**odel Output **E**xtraction

A Python package providing utilities for processing, interpolating, and
converting hydrodynamic ocean model NetCDF datasets.

Why?
----

There are many hydrodynamic modeling frameworks in use today by the
oceanographic community. All models are output to a NetCDF format, but
the structure, dimensions, variables, attributes, naming conventions,
coordinate systems, and masking rules can all vary significantly, making
it difficult for users to extract meaningful information from these
complex systems.

This package attempts to support a common methodology for interacting
with these datasets by abstracting the nuances of each modeling system
into distinct modules and unifying them with a single API.

Supported Models
----------------

*NOTE: To date, this package has been tested to work with National Ocean
Service (NOS) Operational Forecast Systems (OFS) (e.g. CBOFS, DBOFS,
NGOFS, etc.), however with minor tweaks it should also work with output
from non-NOS OFS models.*

The following ocean modeling frameworks are presently supported:

**Regional Ocean Modeling System (ROMS)**

> *\"ROMS is a free-surface, terrain-following, primitive equations
> ocean model widely used by the scientific community for a diverse
> range of applications\...In the horizontal, the primitive equations
> are evaluated using boundary-fitted, orthogonal curvilinear
> coordinates on a staggered Arakawa C-grid. The general formulation of
> curvilinear coordinates includes both Cartesian (constant metrics) and
> spherical (variable metrics) coordinates. Coastal boundaries can also
> be specified as a finite-discretized grid via land/sea masking. As in
> the vertical, the horizontal stencil utilizes a centered, second-order
> finite differences.\"*
>
> Description from <https://www.myroms.org/>

**Finite-Volume Community Ocean Modeling System (FVCOM)**

> *\"FVCOM is a prognostic, unstructured-grid, finite-volume,
> free-surface, 3-D primitive equation coastal ocean circulation model
> developed by UMASSD-WHOI joint efforts. The model consists of
> momentum, continuity, temperature, salinity and density equations and
> is closed physically and mathematically using turbulence closure
> submodels. The horizontal grid is comprised of unstructured triangular
> cells and the irregular bottom is presented using generalized
> terrain-following coordinates.\"*
>
> Description from <http://fvcom.smast.umassd.edu/fvcom/>

**Princeton Ocean Model (POM)**

> *\"POM is a sigma coordinate (terrain-following), free surface ocean
> model with embedded turbulence and wave sub-models, and wet-dry
> capability.\"*
>
> Description from <http://www.ccpo.odu.edu/POMWEB/index.html>

**Hybrid Coordinate Ocean Model (HYCOM)**

> *\"The HYbrid Coordinate Ocean Model is a primitive equation ocean
> general circulation model that evolved from the Miami
> Isopycnic-Coordinate Ocean Model (MICOM) developed by Rainer Bleck and
> colleagues. Vertical coordinates in HYCOM remain isopycnic in the
> open, stratified ocean. However, they smoothly transition to z
> coordinates in the weakly-stratified upper-ocean mixed layer, to
> terrain-following sigma coordinate in shallow water regions, and back
> to level coordinates in very shallow water.\"*
>
> Description from <https://www.hycom.org/attachments/067_overview.pdf>

Additionally, a generic `rectilinear` module has been created to support
any model output whose coordinate system conforms to a rectilinear grid
and whose depth coordinates reflect standard depths/z-levels, however
this module does not support vertical interpolation.

Features
--------

-   Support for ROMS, FVCOM, POM, HYCOM model output
-   Interpolate staggered horizontal coordinates (i.e. ROMS Arakawa-C
    grid rho/eta/xi) to common coordinates before further processing
-   Apply spatially-varying rotation angle to u/v current components to
    obtain true-north/true-east values before further processing
-   Interpolate sigma (bathymetry-following) vertical coordinates to a
    given depth-below-surface, respecting the appropriate vertical
    transformation, if any
-   Given an approximate target grid resolution, generate a regular grid
    definition conforming to the model domain\'s bounding box (or
    optionally to a predefined set of bounding rectangles), and output
    regular grid definition to CF-compliant NetCDF

Model Considerations
--------------------
- All modules in `thyme/models/` are written specifically for
NOAA hydrodynamic ocean models and should be used with caution.
The modules were written by referencing NOS model metadata, NOS model
guidance, and roms, pom, fvcom, and hycom model documentation.

- To develop a custom module to support a new model, use one of the existing modules
(fvcom.py, hycom.py, pom.py, rectilinear.py, or roms,py) as a template, and place
the new python module file alongside the others under the `thyme/model/` folder

- Development of a custom module may be required to support:

    - Model output options specific to your organization
    - Different variable names and dimensions
    - Different vertical coordinates
    - Different variable masks
    - Different vertical or time varying horizontal masks
    - Different date and time format
    - If the model has already been converted to a regular or rectilinear grid
    - If the model output's vertical coordinate system uses standard depth levels versus sigma

Index (Grid Definition) Files
-----------------------------

In order to convert model output files to a regular grid, an index file
must be supplied at runtime. The purpose of the index file is to persist
information that does not change between model runs in order to reduce
the overall processing time per cycle run. The information stored in the
index file includes the output grid definition and general metadata
about the model itself. Index files for FVCOM-based models with a hybrid
(generalized) coordinate system additionally store interpolated vertical
coordinate values for each output grid cell.

When generating a grid definition, the user has a choice between using
the full extent of the model\'s domain or supplying a shapefile
containing one or more polygons defining subgrids
(sub-regions/subdomains) to which the grid definition will conform. If
no subgrid shapefile is specified, the resulting index file will define
a regular grid (with NoData mask) matching the full model domain
(extent). Otherwise, if a subgrid shapefile is specified, the resulting
index file will define a regular grid (with NoData mask) matching the
unified extent of all subgrid polygons that intersect the model domain,
along with information identifying the grid cell ranges that correspond
with each subgrid. Note that when supplying a subgrid shapefile, all
subgrid polygons must be rectangular, congruent with each other, and
adjacent to one another.

A subgrid index is intended to be used to subset the model output into
smaller geographic areas (i.e. tiles), which in turn results in smaller
output file sizes. Optionally, for a subgrid index file, the user may
specify the name of an attribute field within the subgrid shapefile that
uniquely identifies each subgrid (tile). If no field is specified, each
polygon\'s FID value is used as the identifier. This identifier can be
used to construct unique filenames.

Additionally, a land mask polygon shapefile can be supplied when
generating an index file. If supplied, any output grid cells whose
centroid intersects a land polygon will be masked in the final grid
definition.

One index file must be created per ocean forecast system for each
combination of target resolution and extent (whether using the model\'s
full domain extent or subgrid definition).

When to Generate a New Index File
---------------------------------

Once a model index file is created, it can be reused indefinitely for
that model/resolution/extent/land mask until any of those properties
change. For example, if an FVCOM-based model has a hybrid (generalized)
vertical coordinate system that is modified at some point (i.e., sigma
values are changed), any associated index files will need to be
regenerated using a new model output file in the updated format.

Generally, a new index file is required:

-   For each new model
-   For each desired output grid resolution
-   For each desired set of subgrids
-   If the subgrid polygons change
-   If the subgrid attribute identifier changes
-   If the land mask shapefile changes
-   If the underlining model changes (e.g., new geographic extent,
    change to FVCOM sigma coordinates, etc.)

Requirements
------------

This codebase is written for Python 3 and relies on the following python
packages:

-   gdal
-   netCDF4
-   numpy
-   scipy
-   shapely

Installation
------------

The GDAL Python bindings used by this package require system libraries to be
present, so it usually can\'t just be installed using `pip install gdal`.
We recommend installing GDAL either through a package manager (e.g.
`conda`, `apt`, `yum`, `pacman`) or by compiling from scratch.
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is probably
the easiest method.

Once `gdal` has been installed, thyme can be installed using `pip`:

```bash
pip install thyme
```

Example Usage
-------------

To generate a new index file for an FVCOM-based model using the default
grid extent, a \~500 meter target resolution, and a shoreline shapefile
defining land areas to be masked:

```python
from thyme.model import fvcom
native_model_file = fvcom.FVCOMFile('/path/to/existing_fvcom_file.nc')
model_index_file = fvcom.FVCOMIndexFile('/path/to/new_index_file.nc')
try:
  native_model_file.open()
  model_index_file.open()
  model_index_file.init_nc(native_model_file, 500, 'my_fvcom_model', '/path/to/shoreline_shapefile.shp')
finally:
  model_index_file.close()
  native_model_file.close()
```

To generate a new index file for a ROMS-based model using a subgrid
shapefile (with fieldname 'id' used to identify subgrid areas) and a
\~300m target resolution (with no shoreline mask shapefile specified):

```python
from thyme.model import roms
native_model_file = roms.ROMSFile('/path/to/existing_roms_file.nc')
model_index_file = roms.ROMSIndexFile('/path/to/new_index_file.nc')
try:
  native_model_file.open()
  model_index_file.open()
  model_index_file.init_nc(native_model_file, 300, 'my_roms_model', None, '/path/to/subgrid_shapefile.shp', 'subgrid_id_fieldname')
finally:
  model_index_file.close()
  native_model_file.close()
```

To interpolate u/v current components from a ROMS-based model to a
regular grid defined in an existing model index file, at a depth of 4.5
meters below surface, for time index 0, and store the resulting u/v
values in two `numpy.ma.masked_array` objects:

```python
from thyme.model import roms
native_model_file = roms.ROMSFile('/path/to/existing_roms_file.nc')
model_index_file = roms.ROMSIndexFile('/path/to/existing_index_file.nc')
try:
  native_model_file.open()
  model_index_file.open()
  (u_with_mask, v_with_mask) = native_model_file.uv_to_regular_grid(model_index_file, 0, 4.5)
  # u_with_mask and v_with_mask now contain 2D numpy masked arrays
finally:
  model_index_file.close()
  native_model_file.close()
```

Running Tests
-------------

This project uses [pytest](https://docs.pytest.org) for unit testing.

To run the test suite:

```bash
pip install pytest
pytest -vv
```

Authors
-------

-   Erin Nagel (UCAR), <erin.nagel@noaa.gov>
-   Jason Greenlaw (ERT), <jason.greenlaw@noaa.gov>

License
-------

This work, as a whole, is licensed under the BSD 2-Clause License (see
[LICENSE](LICENSE)), however it contains major contributions from the
U.S. National Oceanic and Atmospheric Administration (NOAA), 2017 -
2019, which are individually dedicated to the public domain.

Disclaimer
----------

This repository is a scientific product and is not official
communication of the National Oceanic and Atmospheric Administration, or
the United States Department of Commerce. All NOAA GitHub project code
is provided on an \"as is\" basis and the user assumes responsibility
for its use. Any claims against the Department of Commerce or Department
of Commerce bureaus stemming from the use of this GitHub project will be
governed by all applicable Federal law. Any reference to specific
commercial products, processes, or services by service mark, trademark,
manufacturer, or otherwise, does not constitute or imply their
endorsement, recommendation or favoring by the Department of Commerce.
The Department of Commerce seal and logo, or the seal and logo of a DOC
bureau, shall not be used in any manner to imply endorsement of any
commercial product or activity by DOC or the United States Government.

Acknowledgments
---------------

This software has been developed by the National Oceanic and Atmospheric
Administration (NOAA)/National Ocean Service (NOS)/Office of Coast
Survey (OCS)/Coast Survey Development Lab (CSDL) for use by the
scientific and oceanographic communities.

CSDL wishes to thank the following entities for their assistance:

-   NOAA/NOS/Center for Operational Oceanographic Products and Services
    (CO-OPS)
-   Canadian Hydrographic Service (CHS)
-   Teledyne CARIS
