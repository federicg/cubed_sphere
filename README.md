# cubed_sphere
Build a non-singular unstructured mesh, via gmsh, on a sphere

python requirements,
numpy, scipy, netCDF4, matplotlib, gmsh, meshio

# install first netcdf, with MacOS do
brew install netcdf

the gmsh.py need gmsh bin installed in the pc PATH, 
be sure to have downloaded from https://gmsh.info the Software Development Kit (SDK) 
and to have in the PATH pc variable the gmsh executable


# Download world input dem to be processed at,
http://154.66.220.45:8080/thredds/catalog/regcminput/SURFACE/catalog.html?dataset=regcminput/SURFACE/GTOPO_DEM_30s.nc


#
To visualize type the final mesh type 
gmsh mesh_final.mesh

