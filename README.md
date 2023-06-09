# MWCentreClouds
This repository contains the scripts and data necessary to produce figures similar to those in Noon et al. (2023). The scripts are generalised and are intended to be useful beyond these specific data sets thus the figures will be formatted slightly differently from those featured in Noon et al. 2023.

## Contents
/Radio_cube_utilities.py includes some radio data cube processing tools, including making a moment map and masking a PPV cube. The script also includes a module which produces figure 3 from Noon et al. 2023.

/Radio_maps.py includes modules to produce all other figures from Noon et al. 2023.

/Example.ipynb is a JupyterNotebook which outlines how to use both scripts.

Data/C1_HI_MKT+GBT_COregrid.fits and Data/C2_HI_MKT+GBT_COregrid.fits are the two HI PPV MeerKAT data cubes that have all been feathered with single dish, GBT data. These cubes have been regrid to the spatial resolution of the CO data. For access to C3, please email me as the file is too large to upload to GitHub.

Data/C1_CO_APEX.fits and Data/C2_CO_APEX.fits are the two CO data cubes from the APEX telescope. 

## License
This code is distributed under the terms of the GNU General Public License version 3.0. The text of the license is included in the main directory of the repository as /LICENSE.

## Contact information
If you have any questions, please contact Karlie Noon, karlie.noon@anu.edu.au.
