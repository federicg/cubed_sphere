

import sys
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.spatial import Delaunay
import subprocess
import os
from subprocess import STDOUT, check_output
import netCDF4
import matplotlib.pyplot as plt
import gmsh
import pdb # pdb.set_trace() to set breakpoint

def Sphere2Cube(lon_deg, lat_deg, lat_dem_res, lon_dem_res, idFace, a = 1):

    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    if idFace == 0:
        x = a * np.tan(lon)
        y = a * np.tan(lat) * ( 1. / np.cos(lon) )
        
    elif idFace == 1:
        x = -a * 1. / np.tan(lon)
        y = a * np.tan(lat) * ( 1. / np.sin(lon) )
    
    elif idFace == 2:
        x = a * np.tan(lon)
        y = -a * np.tan(lat) * ( 1. / np.cos(lon) )
    
    elif idFace == 3:
        x = -a * 1. / np.tan(lon)
        y = -a * np.tan(lat) * ( 1. / np.sin(lon) )
    
    elif idFace == 4:
        mask = np.abs(lat)==np.pi/2.
        x = a * np.sin(lon) * 1. / np.tan(lat)
        y = -a * np.cos(lon) * 1. / np.tan(lat)
        
        x[mask] = 0.
        y[mask] = 0.
        
    elif idFace == 5:
        mask = np.abs(lat)==np.pi/2.
        x = -a * np.sin(lon) * 1. / np.tan(lat)
        y = -a * np.cos(lon) * 1. / np.tan(lat)
         
        x[mask] = 0.
        y[mask] = 0.
    else:
        print("Not recognized cube-face Id")
        
        
    return [x, y]
    


def XYZ2LonLat(X, Y, Z, lon_crop_extent, a=1):
    R = a*np.sqrt(3.)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    lat = np.arcsin(Z/r)*180./np.pi
    lon = np.arctan2(Y,X)*180./np.pi
    
    mask = np.logical_and(X==0, Y==0)
    lon[mask] = lon_crop_extent[0]
    
#    lon[np.where(np.isnan(lon))] = lon_crop_extent[0]
    
#    col = lon[lon<lon_crop_extent[0]]
#    inds = np.where(np.isnan(col))
#
#    col_mean = np.nanmean(col)
#    lon[inds] = col_mean
#
#    pdb.set_trace()
    
    return [lat, lon]

def Local2Global(x, y, idFace, a = 1):


    if idFace == 0:
        X = 0*x+a
        Y = x
        Z = y
        
    elif idFace == 1:
        X = -x
        Y = 0*x+a
        Z = y
    
    elif idFace == 2:
        X = 0*x-a
        Y = -x
        Z = y
    
    elif idFace == 3:
        X = x
        Y = 0*x-a
        Z = y
    
    elif idFace == 4:
        X = -y
        Y = x
        Z = 0*x+a
        
    elif idFace == 5:
        X = y
        Y = x
        Z = 0*x-a
         
    else:
        print("Not recognized cube-face Id")
        
        
    return [X, Y, Z]
    
def Global2Local(X, Y, Z, idFace, a = 1):

    if idFace == 0:
        x = a*Y/X
        y = a*Z/X
        
    elif idFace == 1:
        x = -a*X/Y
        y = a*Z/Y
    
    elif idFace == 2:
        x = a*Y/X
        y = -a*Z/X
    
    elif idFace == 3:
        x = -a*X/Y
        y = -a*Z/Y
    
    elif idFace == 4:
        x = a*Y/Z
        y = -a*X/Z
        
    elif idFace == 5:
        x = -a*Y/Z
        y = -a*X/Z
         
    else:
        print("Not recognized cube-face Id")
        
        
    return [x, y]
    
def computeGrad_spherical(z_slice, lat_dem_res, lon_dem_res, LON, LAT, R=np.sqrt(3.)):
    d_z_lat_slice = np.gradient(z_slice, lat_dem_res, axis=0)
    d_z_lon_slice = np.gradient(z_slice, lon_dem_res, axis=1)
    
    d_z_lat = (1./R)*d_z_lat_slice
    d_z_lon =  1./(R*np.cos(np.radians(LAT)))*d_z_lon_slice

    return [d_z_lat, d_z_lon]

def computeGrad(d_z_lat_slice, d_z_lon_slice, LON_deg, LAT_deg, idFace, a=1.):
    """
    compute gradient in (x,y) local cube coordinates starting from (\theta, \lambda) spherical coordinates
    """
    R=a*np.sqrt(3.)
    
    LON = np.radians(LON_deg)
    LAT = np.radians(LAT_deg)
    
    if idFace == 0:
        d_x_lat = 0.
        d_x_lon = a*(1. + np.tan(LON)**2.)
    
        d_y_lat = a*(1. + np.tan(LAT)**2.)/np.cos(LON)
        d_y_lon = a*np.tan(LAT)*np.tan(LON)/np.cos(LON)
        
    elif idFace == 1:
        d_x_lat = 0.
        d_x_lon = a/(np.tan(LON)**2.)*(1. + np.tan(LON)**2.)
    
        d_y_lat = a * (1+np.tan(LAT)**2.) * ( 1 / np.sin(LON) )
        d_y_lon = -a * np.tan(LAT) * ( 1 / (np.sin(LON)**2.) )*np.cos(LON)
        
    
    elif idFace == 2:
        d_x_lat = 0.
        d_x_lon = a*(1+np.tan(LON)**2.)
    
        d_y_lat = -a * (1+np.tan(LAT)**2.) * ( 1 / np.cos(LON) )
        d_y_lon = -a * np.tan(LAT) * ( 1 / (np.cos(LON)**2.) )*np.sin(LON)
        
    
    elif idFace == 3:
        d_x_lat = 0.
        d_x_lon = a*(1+np.tan(LON)**2.)/(np.tan(LON)**2.)
    
        d_y_lat = -a * (1+np.tan(LAT)**2.) * ( 1 / np.sin(LON) )
        d_y_lon = a * np.tan(LAT) * ( 1 / (np.sin(LON)**2.) )*np.cos(LON)
        
    
    elif idFace == 4:
        d_x_lat = -a * np.sin(LON) * 1 / (np.tan(LAT)**2.)*(1+np.tan(LAT)**2.)
        d_x_lon = a * np.cos(LON) * 1 / np.tan(LAT)
    
        d_y_lat = a * np.cos(LON) * 1 / (np.tan(LAT)**2.)*(1+np.tan(LAT)**2.)
        d_y_lon = a * np.sin(LON) * 1 / np.tan(LAT)
        
    elif idFace == 5:
        d_x_lat = a * np.sin(LON) * 1 / (np.tan(LAT)**2.)*(1+np.tan(LAT)**2.)
        d_x_lon = -a * np.cos(LON) * 1 / np.tan(LAT)
    
        d_y_lat = a * np.cos(LON) * 1 / (np.tan(LAT)**2.)*(1+np.tan(LAT)**2.)
        d_y_lon = a * np.sin(LON) * 1 / np.tan(LAT)
         
    else:
        print("Not recognized cube-face Id")
        
    
    d_z_x = d_z_lat_slice*d_x_lat + d_z_lon_slice*d_x_lon
    d_z_y = d_z_lat_slice*d_y_lat + d_z_lon_slice*d_y_lon
    
    
    return [d_z_x, d_z_y]
    
    

    
def compute_size_field(nodes, triangles, oro_interp, d_z_lat_interp, d_z_lon_interp, starting_mesh_size, lat_crop_extent, lon_crop_extent, a=1):
    
    
    tau = 0.1
    
    xyz = nodes[triangles]
    
    lat_interp_node, lon_interp_node = XYZ2LonLat(nodes[:,0], nodes[:,1], nodes[:,2], lon_crop_extent, a)
    
    z_node = oro_interp((lat_interp_node, lon_interp_node), method='linear')
    
    
    
#    plt.plot(lon_interp_node, lat_interp_node, 'o')
#    plt.axis('equal')
#    plt.show()
    
#    plt.triplot(x_local, y_local, triangles)
#    plt.tricontourf(x_local, y_local, triangles, z_node)
    plt.tricontourf(lon_interp_node, lat_interp_node, triangles, z_node)
    plt.axis('equal')
    plt.colorbar()
    plt.show()
    
    
    z_node = z_node[triangles]
    
    xyz_middle = (xyz[:,0,:] + xyz[:,1,:] + xyz[:,2,:])/3.
    lat_interp_middle, lon_interp_middle = XYZ2LonLat(xyz_middle[:,0], xyz_middle[:,1], xyz_middle[:,2], lon_crop_extent, a)
    d_z_lat_middle = d_z_lat_interp((lat_interp_middle, lon_interp_middle), method='nearest')
    d_z_lon_middle = d_z_lon_interp((lat_interp_middle, lon_interp_middle), method='nearest')
    
    
    
    N_elements = xyz.shape[0]
    sf = np.zeros(xyz.shape[0])
    z_normal = np.array([0.,0.,1.])
    
    iCubeFace = -1
    for iTri in range(xyz.shape[0]):
        xyz_triangle = xyz[iTri]
        z_node_triangle = z_node[iTri]
        
        
        z_grad = np.array([0.,0.,0.])
        for iEdge in range(-2,1):
            edge_normal = np.cross(z_normal, xyz_triangle[iEdge,:] - xyz_triangle[iEdge+1,:])
            
            if np.dot(edge_normal, xyz_triangle[iEdge+2,:] - xyz_triangle[iEdge,:]) < 0:
                edge_normal *= -1.
            
            
            z_grad += edge_normal * z_node_triangle[ iEdge+2 ]
            
        
#        area_triangle = np.cross(xyz_triangle[1,:] - xyz_triangle[0,:], xyz_triangle[2,:] - xyz_triangle[0,:])
#        area_triangle = np.linalg.norm(area_triangle)/2.
        
        
        if (np.allclose(xyz_triangle[0,0],1.) and np.allclose(xyz_triangle[1,0],1.) and np.allclose(xyz_triangle[2,0],1.)):
            iCubeFace = 0
        elif (np.allclose(xyz_triangle[0,1],1.) and np.allclose(xyz_triangle[1,1],1.) and np.allclose(xyz_triangle[2,1],1.)):
            iCubeFace = 1
        elif (np.allclose(xyz_triangle[0,0],-1.) and np.allclose(xyz_triangle[1,0],-1.) and np.allclose(xyz_triangle[0,0],-1.)):
            iCubeFace = 2
        elif (np.allclose(xyz_triangle[0,1],-1.) and np.allclose(xyz_triangle[1,1],-1.) and np.allclose(xyz_triangle[2,1],-1.)):
            iCubeFace = 3
        elif (np.allclose(xyz_triangle[0,2],1.) and np.allclose(xyz_triangle[1,2],1.) and np.allclose(xyz_triangle[2,2],1.)):
            iCubeFace = 4
        elif (np.allclose(xyz_triangle[0,2],-1.) and np.allclose(xyz_triangle[1,2],-1.) and np.allclose(xyz_triangle[2,2],-1.)):
            iCubeFace = 5
        else:
            print("Not recognized cube-face Id")
            
            
            
        d_z_x_middle, d_z_y_middle = computeGrad(d_z_lat_middle[iTri], d_z_lon_middle[iTri], lon_interp_middle[iTri], lat_interp_middle[iTri], iCubeFace, a)
        eta_k = np.linalg.norm(z_grad - np.array([d_z_x_middle, d_z_y_middle, 0.]))
        
        candidate = np.sqrt(1./N_elements/.5 * (tau/(eta_k+1.e-8))**2.)
 
        sf[ iTri ] = np.maximum(np.minimum(candidate, starting_mesh_size*4), starting_mesh_size/2) #np.maximum(np.minimum(candidate, starting_mesh_size*2), starting_mesh_size/2)
        
    
    return sf
    

def GetId(LatOrLonDEM, LatOrLonExtent, LatLon_dem_res):
    id = int(((LatOrLonExtent - LatOrLonDEM)/LatLon_dem_res))
    return id

class Mesh:
    def __init__(self):
        [self.vtags, vxyz, _] = gmsh.model.mesh.getNodes()
        
        self.vxyz = vxyz.reshape((-1, 3))
        vmap = dict({j: i for i, j in enumerate(self.vtags)})
        self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))



def modifyMeshFile(file, output_file, idFace, a=1):
    # back-transform in the coordinates of the sphere
    #
    file_o = open(file)
    listOfLines = file_o.readlines()

    #
    file = open(output_file,"w")
    

    for i in range(len(listOfLines)):
        line = listOfLines[i]
      
        if line.strip() == "Vertices":
            nbVertices = listOfLines[i+1]
            file.write("MeshVersionFormatted 2\n")
            file.write("Dimension\n")
            file.write("3\n")
            file.write("Vertices\n")
            file.write(nbVertices)# + "\n")
            for j in range(int(nbVertices)):
                k = j+i+1+1
                
                myarray = listOfLines[k].split()
                x = float(myarray[0])
                y = float(myarray[1])
                #z = myarray[2]
                [lat, lon] = Cube2Sphere(x, y, idFace, a)
                file.write(str(lon) + " " + str(lat) + " " + str(a) + " " + myarray[3] + "\n")
                #print(myarray[0] + " " + myarray[1])
                
            i = k

        if line.strip() == "Quadrilaterals":
            nbQuad = listOfLines[i+1]
            file.write("Quadrilaterals\n")
            file.write(nbQuad)
            for j in range(int(nbQuad)):
                k = j+i+1+1

                myarray = listOfLines[k].split()
                file.write(myarray[0] + " " + myarray[1] + " " + myarray[2] + " " + myarray[3] + " " + myarray[4] + "\n")
                #file.write(listOfLines[k])
                
    
    file.write("End")
        
    file_o.close()
    file.close()
    
def writeDatFileQuad(meshFile, output_file):

    #
    file_o = open(meshFile,"r")
    listOfLines = file_o.readlines()

    #
    file = open(output_file,"w")



    for i in range(len(listOfLines)):
        line = listOfLines[i]
      
        if line.strip() == "Vertices":
            nbVertices = listOfLines[i+1]
            file.write(nbVertices)# + "\n")
            for j in range(int(nbVertices)):
                k = j+i+1+1
                
                myarray = listOfLines[k].split()
                file.write(myarray[0] + " " + myarray[1] + " " + myarray[2] + "\n")
                #print(myarray[0] + " " + myarray[1])
            i = k

        if line.strip() == "Quadrilaterals":
            nbHex = listOfLines[i+1]
            file.write(nbHex)
            for j in range(int(nbHex)):
                k = j+i+1+1

                myarray = listOfLines[k].split()
                file.write(myarray[4] + " " + myarray[0] + " " + myarray[1] + " " + myarray[2] + " " + myarray[3] + "\n")
                #file.write(listOfLines[k])

    file_o.close()
    file.close()
            
            

###################################
## Inputs

# lon, [30, 50]
# lat, [30, 40]

## Longitude coordinates of the desired slice of the orography  [-180, 180)
lon_crop_extent = [30, 50] #[-180, -150.0]#[-10, 10.0]#[-181, -179]#[-179, -150.0]#[-180, -150.0] #[-13.5, 50.0] #[-190, -170.0] # [-190, -170] # thanks to Python that admits negative ids
## Latitude coordinates of the desired slice of the orography [-90, 90]
lat_crop_extent = [30, 40] #[46, 64.6]#[20, 30.6]#[46, 64.6]#[67, 69]#[46, 60] #[27.7, 64.6] #[-10, 10]

#[50, 60]
###################################




## read orography file
DEM = netCDF4.Dataset('GTOPO_DEM_30s.nc','r')
lon_dem =  DEM.variables['lon'][:]
lat_dem =  DEM.variables['lat'][:]
z_dem   =  DEM.variables['z'][:][:]




# intial data resolution (0.0083333333333 deg)
lon_dem_res = (lon_dem[-1] - lon_dem[0]) / np.size(lon_dem) # fine res.
lat_dem_res = (lat_dem[-1] - lat_dem[0]) / np.size(lat_dem)




# compute matrix index of the slice
lon_id = [GetId(lon_dem[ 0 ], lon_crop_extent[ 0 ], lon_dem_res), GetId(lon_dem[ 0 ], lon_crop_extent[ 1 ], lon_dem_res)]
lat_id = [GetId(lat_dem[ 0 ], lat_crop_extent[ 0 ], lat_dem_res), GetId(lat_dem[ 0 ], lat_crop_extent[ 1 ], lat_dem_res)]



# load desired slices
print("Loading data...")
lon_slice = lon_dem[range(lon_id[0], lon_id[1])]
lat_slice = lat_dem[range(lat_id[0], lat_id[1])]
z_slice   = z_dem[  lat_id[0]: lat_id[1], \
                    range(lon_id[0], lon_id[1])]
print("Data loaded")
DEM.close()

# z_slice.shape, lon_slice.shape, lat_slice.shape


# divide z, lon_slice, lat_slice in at max 6 slices, 6 cube faces
# following paper nair_et_al_2005

#
tri = [[] for _ in range(6)]



isFaceP1 = False
isFaceP2 = False
isFaceP3 = False
isFaceP4 = False
isFaceP5 = False
isFaceP6 = False


lat_slice = lat_slice.data
lon_slice = lon_slice.data
z_slice   = z_slice.data

if (lon_slice[0] > lon_slice[-1]): # non ascending order, pb with RegularGridInterpolator
    lon_slice[lon_slice >= lon_slice[0]] -= 360.

LON, LAT = np.meshgrid(lon_slice, lat_slice)

plt.contourf(LON, LAT, z_slice)
plt.axis('equal')
plt.colorbar()
plt.show()




# structured grid, faster than stadard scipy interpolation
z_mesh_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), z_slice, fill_value=None, bounds_error=False, method='linear')
d_z_lat, d_z_lon = computeGrad_spherical(z_slice, lat_dem_res, lon_dem_res, LON, LAT)
d_z_lat_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), d_z_lat, fill_value=None, bounds_error=False, method='nearest')
d_z_lon_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), d_z_lon, fill_value=None, bounds_error=False, method='nearest')






# add vertex/intersections
Nl = 100
Nt = 100
lon_slice_mod = np.linspace(lon_crop_extent[0], lon_crop_extent[-1], Nl)
lat_slice_mod = np.linspace(lat_crop_extent[0], lat_crop_extent[-1], Nt)

if (45>lon_crop_extent[0] and 45<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, 45)
    
if (-45>lon_crop_extent[0] and -45<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, -45)
    
if (135>lon_crop_extent[0] and 135<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, 135)
    
if (-135>lon_crop_extent[0] and -135<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, -135)
    
if (-225>lon_crop_extent[0] and -225<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, -225)
    
if (-315>lon_crop_extent[0] and -315<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, -315)

candidate = np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    

candidate = np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arccos(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
    
candidate = -np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arccos(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
    
candidate = -np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)

candidate = -np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)

candidate = -np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = -np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[0]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi - 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)
    
candidate = np.arcsin(np.tan(lat_crop_extent[-1]*np.pi/180.))*180./np.pi + 180.
if (candidate>lon_crop_extent[0] and candidate<lon_crop_extent[-1]):
    lon_slice_mod = np.append(lon_slice_mod, candidate)


lon_slice_mod = np.unique(lon_slice_mod)
 
 
lat_app_candidate = np.arctan(-np.cos(lon_slice_mod*np.pi/180.))*180./np.pi
lat_app_candidate = lat_app_candidate[np.logical_and(lat_app_candidate>=lat_crop_extent[0], lat_app_candidate<=lat_crop_extent[-1])]
lat_slice_mod = np.append(lat_slice_mod, lat_app_candidate)


lat_app_candidate = np.arctan( np.cos(lon_slice_mod*np.pi/180.))*180./np.pi
lat_app_candidate = lat_app_candidate[np.logical_and(lat_app_candidate>=lat_crop_extent[0], lat_app_candidate<=lat_crop_extent[-1])]
lat_slice_mod = np.append(lat_slice_mod, lat_app_candidate)

lat_app_candidate = np.arctan(-np.sin(lon_slice_mod*np.pi/180.))*180./np.pi
lat_app_candidate = lat_app_candidate[np.logical_and(lat_app_candidate>=lat_crop_extent[0], lat_app_candidate<=lat_crop_extent[-1])]
lat_slice_mod = np.append(lat_slice_mod, lat_app_candidate)

lat_app_candidate = np.arctan( np.sin(lon_slice_mod*np.pi/180.))*180./np.pi
lat_app_candidate = lat_app_candidate[np.logical_and(lat_app_candidate>=lat_crop_extent[0], lat_app_candidate<=lat_crop_extent[-1])]
lat_slice_mod = np.append(lat_slice_mod, lat_app_candidate)


lat_slice_mod = np.unique(lat_slice_mod)

LON_mod, LAT_mod = np.meshgrid(lon_slice_mod, lat_slice_mod)


mask_crop_boundary = LON_mod==LON_mod
mask_crop_boundary = mask_crop_boundary.astype(np.int) - ndimage.binary_erosion(mask_crop_boundary.astype(np.int))
mask_crop_boundary = mask_crop_boundary.astype(np.bool)




mask_crop_boundary_mod = mask_crop_boundary.astype(np.int)
if (mask_crop_boundary_mod[LON_mod==  45].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod==  45, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod==  45, LAT_mod==lat_slice_mod[-1])]  += 1
    
if (mask_crop_boundary_mod[LON_mod== -45].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod== -45, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod== -45, LAT_mod==lat_slice_mod[-1])]  += 1

if (mask_crop_boundary_mod[LON_mod== 135].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod== 135, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod== 135, LAT_mod==lat_slice_mod[-1])]  += 1
    
if (mask_crop_boundary_mod[LON_mod==-135].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod== -135, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod== -135, LAT_mod==lat_slice_mod[-1])]  += 1
    
if (mask_crop_boundary_mod[LON_mod==-225].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod== -225, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod== -225, LAT_mod==lat_slice_mod[-1])]  += 1

if (mask_crop_boundary_mod[LON_mod==-315].size>0):
    mask_crop_boundary_mod[np.logical_and(LON_mod== -315, LAT_mod==lat_slice_mod[0 ])]  += 1
    mask_crop_boundary_mod[np.logical_and(LON_mod== -315, LAT_mod==lat_slice_mod[-1])]  += 1


toll = 1e-10 # tolerance for round-off errors

mask_crop_boundary_mod[np.abs(LAT_mod-np.arctan(-np.cos(LON_mod*np.pi/180.))*180./np.pi)<toll] += 1
mask_crop_boundary_mod[np.abs(LAT_mod-np.arctan( np.cos(LON_mod*np.pi/180.))*180./np.pi)<toll] += 1
mask_crop_boundary_mod[np.abs(LAT_mod-np.arctan(-np.sin(LON_mod*np.pi/180.))*180./np.pi)<toll] += 1
mask_crop_boundary_mod[np.abs(LAT_mod-np.arctan( np.sin(LON_mod*np.pi/180.))*180./np.pi)<toll] += 1


mask_crop_boundary_mod[0 , 0] += 1
mask_crop_boundary_mod[-1, 0] += 1
mask_crop_boundary_mod[0 ,-1] += 1
mask_crop_boundary_mod[-1,-1] += 1

plt.plot(LON_mod[mask_crop_boundary_mod>0], LAT_mod[mask_crop_boundary_mod>0], 'o')
plt.axis('equal')
plt.show()


mask_crop_boundary_mod = mask_crop_boundary_mod>1


plt.plot(LON_mod[mask_crop_boundary_mod], LAT_mod[mask_crop_boundary_mod], 'o')
plt.axis('equal')
plt.show()




    



mask = np.logical_and(np.abs(LAT)<np.arctan(np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)>1./np.sqrt(2.))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP1 = True
    
    print("isFaceP1 ", isFaceP1)
    
    bool_val = np.logical_not(np.logical_and(np.abs(LAT_mod)<np.arctan(np.cos(LON_mod*np.pi/180.))*180./np.pi+toll, np.cos(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll))
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    
    
#    plt.plot(LON[mask], LAT[mask], 'o')
#    plt.axis('equal')
#    plt.show()
    
    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    
    
    points = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
    
    
    tri[0] = Delaunay(points)
    
    plt.triplot(tri[0].points[:,0], tri[0].points[:,1], tri[0].simplices)
    plt.plot(tri[0].points[:,0], tri[0].points[:,1], 'o')
    plt.axis('equal')
    plt.show()


mask = np.logical_and(np.abs(LAT)<np.arctan(np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)>1./np.sqrt(2.))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP2 = True
    
    print("isFaceP2 ", isFaceP2)
    
    bool_val = np.logical_not(np.logical_and(np.abs(LAT_mod)<np.arctan(np.sin(LON_mod*np.pi/180.))*180./np.pi+toll, np.sin(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll))
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    

#    plt.plot(LON[mask], LAT[mask], 'o')
#    plt.axis('equal')
#    plt.show()
    
    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    
    points  = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
        
    
    tri[1] = Delaunay(points)
        
    

    plt.triplot(tri[1].points[:,0], tri[1].points[:,1], tri[1].simplices)
    plt.plot(tri[1].points[:,0], tri[1].points[:,1], 'o')
    plt.axis('equal')
    plt.show()
    
    
    
mask = np.logical_and(np.abs(LAT)<-np.arctan(np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)<-1./np.sqrt(2.))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP3 = True
    
    print("isFaceP3 ", isFaceP3)
    
    bool_val = np.logical_not(np.logical_and(np.abs(LAT_mod)<-np.arctan(np.cos(LON_mod*np.pi/180.))*180./np.pi+toll, np.cos(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    
    
#    plt.plot(LON[mask], LAT[mask], 'o')
#    plt.axis('equal')
#    plt.show()

    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    
    
    points  = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
            
  
    tri[2] = Delaunay(points)

    plt.triplot(tri[2].points[:,0], tri[2].points[:,1], tri[2].simplices)
    plt.plot(tri[2].points[:,0], tri[2].points[:,1], 'o')
    plt.axis('equal')
    plt.show()
    
    
mask = np.logical_and(np.abs(LAT)<-np.arctan(np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)<-1./np.sqrt(2.))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP4 = True
    
    print("isFaceP4 ", isFaceP4)
    
    bool_val = np.logical_not(np.logical_and(np.abs(LAT_mod)<-np.arctan(np.sin(LON_mod*np.pi/180.))*180./np.pi+toll, np.sin(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    
    
    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    
    
    points  = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
        
    tri[3] = Delaunay(points)

    plt.triplot(tri[3].points[:,0], tri[3].points[:,1], tri[3].simplices)
    plt.plot(tri[3].points[:,0], tri[3].points[:,1], 'o')
    plt.axis('equal')
    plt.show()


mask = np.logical_and(LAT>np.arctan(np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)>1./np.sqrt(2.))
mask = np.logical_or(mask, np.logical_and(LAT>np.arctan(np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)>1./np.sqrt(2.)))
mask = np.logical_or(mask, np.logical_and(LAT>np.arctan(-np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)<-1./np.sqrt(2.)))
mask = np.logical_or(mask, np.logical_and(LAT>np.arctan(-np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)<-1./np.sqrt(2.)))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP5 = True
    
    print("isFaceP5 ", isFaceP5)
    
    bool_val = np.logical_and(LAT_mod>np.arctan(np.cos(LON_mod*np.pi/180.))*180./np.pi-toll, np.cos(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll)
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod>np.arctan(np.sin(LON_mod*np.pi/180.))*180./np.pi-toll, np.sin(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll))
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod>np.arctan(-np.cos(LON_mod*np.pi/180.))*180./np.pi-toll, np.cos(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod>np.arctan(-np.sin(LON_mod*np.pi/180.))*180./np.pi-toll, np.sin(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))


    bool_val = np.logical_not(bool_val)
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    
    
    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    points  = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
        
        
    tri[4] = Delaunay(points)

    plt.triplot(tri[4].points[:,0], tri[4].points[:,1], tri[4].simplices)
    plt.plot(tri[4].points[:,0], tri[4].points[:,1], 'o')
    plt.axis('equal')
    plt.show()
    
    

mask = np.logical_and(LAT<np.arctan(-np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)>1./np.sqrt(2.))
mask = np.logical_or(mask, np.logical_and(LAT<np.arctan(-np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)>1./np.sqrt(2.)))
mask = np.logical_or(mask, np.logical_and(LAT<np.arctan(np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)<-1./np.sqrt(2.)))
mask = np.logical_or(mask, np.logical_and(LAT<np.arctan(np.sin(LON*np.pi/180.))*180./np.pi, np.sin(LON*np.pi/180)<-1./np.sqrt(2.)))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP6 = True

    print("isFaceP6 ", isFaceP6)

    
    bool_val = np.logical_and(LAT_mod<np.arctan(-np.cos(LON_mod*np.pi/180.))*180./np.pi+toll, np.cos(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll)
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod<np.arctan(-np.sin(LON_mod*np.pi/180.))*180./np.pi+toll, np.sin(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll))
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod<np.arctan(np.cos(LON_mod*np.pi/180.))*180./np.pi+toll, np.cos(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))
    bool_val = np.logical_or(bool_val, np.logical_and(LAT_mod<np.arctan(np.sin(LON_mod*np.pi/180.))*180./np.pi+toll, np.sin(LON_mod*np.pi/180)<-1./np.sqrt(2.)+toll))
    

    bool_val = np.logical_not(bool_val)
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(np.bool)
    mask_crop_boundary_p[bool_val] = False
    
#    plt.plot(LON[mask], LAT[mask], 'o')
#    plt.axis('equal')
#    plt.show()
    
    plt.plot(LON_mod[mask_crop_boundary_p], LAT_mod[mask_crop_boundary_p], 'o')
    plt.axis('equal')
    plt.show()
    
    points  = []
    for ii in range(LON_mod[mask_crop_boundary_p].size):
        p1 = [LON_mod[mask_crop_boundary_p][ii], LAT_mod[mask_crop_boundary_p][ii]]
        points.append( p1 )
        

    tri[5] = Delaunay(points)

    plt.triplot(tri[5].points[:,0], tri[5].points[:,1], tri[5].simplices)
    plt.plot(tri[5].points[:,0], tri[5].points[:,1], 'o')
    plt.axis('equal')
    plt.show()





isFace_vect = [isFaceP1, isFaceP2, isFaceP3, isFaceP4, isFaceP5, isFaceP6]

starting_mesh_size = 0.0028


for i in range(6):
    if isFace_vect[ i ]:
        plt.triplot(tri[i].points[:,0], tri[i].points[:,1], tri[i].simplices)
        plt.plot(tri[i].points[:,0], tri[i].points[:,1], 'o')

plt.axis('equal')
plt.show()




# build sf_view
gmsh.initialize()
gmsh.option.setNumber("Mesh.RecombineAll", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.Format", 30)
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.SaveElementTagType", 3)

current_model = "starting"
gmsh.model.add(current_model)

pointPhy_globalList = []
pointVertex_globalList = []
e_globalList = []
eList_full = []
index = 1
iLoop = 1
iSurf = 1
current_e = 1
for ii in range(6):
    if isFace_vect[ ii ]:
    


        for jj in range(tri[ii].simplices.shape[0]):
        
            
            poi = tri[ii].points[tri[ii].simplices][jj]
            
            
            eList = []
            for iVert in range(3):
                p1 = poi[iVert-1]
                p2 = poi[iVert  ]
            
                
                bool_val1 = True
                bool_val2 = True
                for point in pointVertex_globalList:
                    if (np.allclose([p1, p2],point)):
                        bool_val1 = False
                    if (np.allclose([p2, p1],point)):
                        bool_val2 = False
                
                
                if (bool_val1 and bool_val2):
                    
            
                    XX, YY = Sphere2Cube([p1[0], p2[0]], [p1[1], p2[1]], lat_dem_res, lon_dem_res, ii)
                    X_gl, Y_gl, Z_gl = Local2Global(XX, YY, ii)
                    
                    
                    
            
                    pList = []
                    for kk in range(np.size(X_gl)):
                        candidate = [np.allclose([X_gl[kk], Y_gl[kk], Z_gl[kk]], point) for point in pointPhy_globalList]
                        
                        if (not(np.sum(candidate)>0) or np.size(candidate)==0):
                            gmsh.model.geo.addPoint(X_gl[kk], Y_gl[kk], Z_gl[kk], meshSize=starting_mesh_size, tag=index)
                            pointPhy_globalList.append([X_gl[kk], Y_gl[kk], Z_gl[kk]])
                            pList.append(index)
                            index += 1
                        else:
                            pList.append(candidate.index(True)+1)
                    
                    
                    
                    gmsh.model.geo.addLine(pList[0], pList[-1], tag=current_e)
                    e_globalList.append(current_e)
                    e_globalList.append(-current_e)
                    
                    eList.append(current_e)
                    eList_full.append(current_e)
                    
                    pointVertex_globalList.append( [p1, p2] )
                    pointVertex_globalList.append( [p2, p1] )
                    
                    current_e += 1
                    
                    
                else:
                    
                    candidate = [np.allclose([p1, p2],point) for point in pointVertex_globalList].index(True)
                    eList.append(e_globalList[candidate])
                    
            
          
            gmsh.model.geo.addCurveLoop(eList, iLoop)
            gmsh.model.geo.addPlaneSurface([iLoop], iSurf)
            iLoop += 1
            iSurf += 1
            
        
        


gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, eList_full, 1)
gmsh.model.mesh.generate(2)
gmsh.write("mesh_starting.msh")
            
mesh = Mesh()
            
print("mesh elements, ", mesh.triangles.shape[0])
            
nodes_coords = np.reshape(gmsh.model.mesh.getNodesForPhysicalGroup(1,1)[1],(-1,3))
nodes_ids    = gmsh.model.mesh.getNodesForPhysicalGroup(1,1)[0]
            

sf_ele = compute_size_field(mesh.vxyz, mesh.triangles, z_mesh_fem, d_z_lat_fem, d_z_lon_fem, starting_mesh_size, lat_crop_extent, lon_crop_extent)
            
sf_node = nodes_ids/nodes_ids*starting_mesh_size
            
sf_view = gmsh.view.add("mesh size field")
gmsh.view.addModelData(sf_view, 0, current_model, "ElementData", mesh.triangles_tags, sf_ele[:, None])
gmsh.view.addModelData(sf_view, 0, current_model, "NodeData", nodes_ids, sf_node[:,None])
gmsh.view.write(sf_view, "sf.pos")
        

# Force gmsh to use just the background mesh provided
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

current_model = "final"
gmsh.model.add(current_model)


pointPhy_globalList = []
pointVertex_globalList = []
e_globalList = []
index = 1
iLoop = 1
iSurf = 1
current_e = 1
for ii in range(6):
    if isFace_vect[ ii ]:
    


        for jj in range(tri[ii].simplices.shape[0]):
        
            
            poi = tri[ii].points[tri[ii].simplices][jj]
            
            
            eList = []
            for iVert in range(3):
                p1 = poi[iVert-1]
                p2 = poi[iVert  ]
            
                
                bool_val1 = True
                bool_val2 = True
                for point in pointVertex_globalList:
                    if (np.allclose([p1, p2],point)):
                        bool_val1 = False
                    if (np.allclose([p2, p1],point)):
                        bool_val2 = False
                
                
                if (bool_val1 and bool_val2):
                    
            
                    XX, YY = Sphere2Cube([p1[0], p2[0]], [p1[1], p2[1]], lat_dem_res, lon_dem_res, ii)
                    X_gl, Y_gl, Z_gl = Local2Global(XX, YY, ii)
                    
            
                    pList = []
                    for kk in range(np.size(X_gl)):
                        candidate = [np.allclose([X_gl[kk], Y_gl[kk], Z_gl[kk]], point) for point in pointPhy_globalList]
                        
                        if (not(np.sum(candidate)>0) or np.size(candidate)==0):
                            gmsh.model.geo.addPoint(X_gl[kk], Y_gl[kk], Z_gl[kk], meshSize=starting_mesh_size, tag=index)
                            pointPhy_globalList.append([X_gl[kk], Y_gl[kk], Z_gl[kk]])
                            pList.append(index)
                            index += 1
                        else:
                            pList.append(candidate.index(True)+1)
            
                    
                    gmsh.model.geo.addLine(pList[0], pList[-1], tag=current_e)
                    e_globalList.append(current_e)
                    e_globalList.append(-current_e)
                    
                    eList.append(current_e)
                    
                    pointVertex_globalList.append( [p1, p2] )
                    pointVertex_globalList.append( [p2, p1] )
                    
                    current_e += 1
                    
                    
                else:
                    
                    candidate = [np.allclose([p1, p2],point) for point in pointVertex_globalList].index(True)
                    eList.append(e_globalList[candidate])
                    
            
          
            gmsh.model.geo.addCurveLoop(eList, iLoop)
            gmsh.model.geo.addPlaneSurface([iLoop], iSurf)
            iLoop += 1
            iSurf += 1
            
            
gmsh.model.geo.synchronize()

bg_field = gmsh.model.mesh.field.add("PostView")
gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", sf_view)
gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
gmsh.model.mesh.generate(2)
gmsh.write("mesh_final.msh")

#gmsh.write("mesh_final.mesh")
#modifyMeshFile("mesh_final.mesh", "mesh_mod.mesh", ii)
              






gmsh.finalize()
sys.exit()






sys.exit()

for ii in range(0,6):
    isFace = isFace_vect[ ii ]
    if isFace:
        file_o = open("mesh2.mesh","r")
        





os.remove("*.mesh")




sys.exit()








