# cubed sphere mesh adaptation procedure 
Build a non-singular unstructured mesh, via gmsh, on a sphere

python requirements,
numpy, scipy, netCDF4, matplotlib, gmsh, meshio

install first netcdf, with MacOS do
brew install netcdf

the gmsh.py need gmsh bin installed in the pc PATH, 
be sure to have downloaded from https://gmsh.info the Software Development Kit (SDK) 
and to have in the PATH pc variable the gmsh executable


Download world input dem to be processed at,
http://sgbd.acmad.org:8080/thredds/fileServer/regcminput/SURFACE/GTOPO_DEM_30s.nc




Apart from the DEM, the input data to the code must be specified between lines 333-349.
Below the explanation of the input data:
- lon_crop_extent: extent in the longitude spherical coordinate
- lat_crop_extent: extent in the latitude spherical coordinate
- earth_radius: Earth radius
- MeshAlgorithm: is the integer that specifies the meshing algorithm adopted, the number corresponds to the one presented in the gmsh guide. Here is the explanation taken from gmsh guide,
"2D mesh algorithm (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad)"
- starting_mesh_size: is the initial mesh size to mesh the cubed domain
- delta_min: is the minimum spatial mesh resolution when adapting in the cubed domain
- delta_max: is the maximum spatial mesh resolution when adapting in the cubed domain

The output mesh is stored in a file name "mesh_final_cube.mesh". This file serves as input to numerical codes that solve the PDE in spherical coordinates by projecting on the Cartesian cubed domain. Refer to (Sadourny 1972, Nair et al. 2015) for an explanation on how transform the continuity equation on the cubical domain.
While in the file "mesh_final.mesh" it is saved the corresponding mesh in longitude-latitude coordinates.

