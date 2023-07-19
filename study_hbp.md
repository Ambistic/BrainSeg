This document describes the protocol used to evaluate the difference
between the automatic registration and manual registration of
the fluorescent slide on its Nissl "counterpart".


Slides selected are located in imagesczi_2 monkey M173 and slides :
- 075
- 083
- 091
- 099

Nissl is in the "CV" folder, fluo is in the "Shading corrected serie SBRI"
folder

Fluo contour is made with the pixel classifier from Pierre (easy to do).
A point is added to be used as comparison in the later registration.

Nissl contours are computed using the "apply_qupath_version1.py" file
first and then the "convert_png_svg_qupath_version1.py" on its outputs.

# HERE

- create the registration script
- run its with proper argument
- do a manual registration
- export the geojson file
- compare the position of the landmark in µm
- write the report




Acted on git commit number : XXX