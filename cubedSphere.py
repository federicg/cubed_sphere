######!/usr/bin/python3.8

import sys
import numpy as np
from scipy import interpolate
#from scipy import optimize
#from scipy import griddata
import subprocess
import os
from subprocess import STDOUT, check_output
import netCDF4
import matplotlib.pyplot as plt
import gmsh


def Sphere2Cube(lon, lat, idFace, a = 1):

    if idFace == 0:
        x = a * np.tan(lon)
        y = a * np.tan(lat) * ( 1 / np.cos(lon) )
    elif idFace == 1:
        x = -a * 1 / np.tan(lon)
        y = a * np.tan(lat) * ( 1 / np.sin(lon) )
    
    elif idFace == 2:
        x = a * np.tan(lon)
        y = -a * np.tan(lat) * ( 1 / np.cos(lon) )
    
    elif idFace == 3:
        x = -a * 1 / np.tan(lon)
        y = -a * np.tan(lat) * ( 1 / np.sin(lon) )
    
    elif idFace == 4:
        x = a * np.sin(lon) * 1 / np.tan(lat)
        y = -a * np.cos(lon) * 1 / np.tan(lat)
        
    elif idFace == 5:
         x = -a * np.sin(lon) * 1 / np.tan(lat)
         y = -a * np.cos(lon) * 1 / np.tan(lat)
         
    else:
        print("Not recognized Id")
        
        
    return [x,y]

            
def Cube2Sphere(x, y, idFace, a=1):

    if idFace == 0:
        lon = np.arctan2(x,a)
        lat = np.arctan(y * np.cos(lon) / a)
        
    elif idFace == 1:
        lon = np.arctan2(-a,x)
        lat = np.arctan(y * np.sin(lon) / a)
    
    elif idFace == 2:
        lon = np.arctan2(x,a)
        lat = np.arctan(-y * np.cos(lon) / a)
    
    elif idFace == 3:
        lon = np.arctan2(-a,x)
        lat = np.arctan(-y * np.sin(lon) / a)
    
    elif idFace == 4:
        lon = np.arctan2(-x,y)
        lat = np.arctan(np.sin(lon) * a/x)
        
    elif idFace == 5:
        lon = np.arctan2(x,y)
        lat = np.arctan(-np.sin(lon) * a/x)
         
    else:
        print("Not recognized Id")
        
        
    return [lat, lon]



def triangle_max_edge(x):
    a = np.sum((x[:, 0, :] - x[:, 1, :])**2, 1)**0.5
    b = np.sum((x[:, 0, :] - x[:, 2, :])**2, 1)**0.5
    c = np.sum((x[:, 1, :] - x[:, 2, :])**2, 1)**0.5
    return np.maximum(a, np.maximum(b, c))

def grad_func(xyz, z, XX, YY):
    #a = 6 * (np.hypot(xyz[..., 0] - .5, xyz[..., 1] - .5) - .2)
    #f = np.real(np.arctanh(a + 0j))
    
    
    #compute the gradient over z direction
    z_grad = np.gradient(z)  # z_grad[0], z_grad[1], each of length equal to length z
    
    #print(np.shape(xyz[...,0]))
    #points_interp = np.random.rand(np.size(xyz[...,0]), 2)
    #print(points_interp)
    #points_interp[:,0] = xyz[...,0]
    #points_interp[:,1] = xyz[...,1]
    
    points = np.random.rand(np.size(XX), 2)
    points[:,0] = np.reshape(XX, np.size(XX))
    points[:,1] = np.reshape(YY, np.size(YY))

    #print(XX)
    #print(z_grad[0])
    #print(np.size(z_grad[0]))
    #print(np.size(points))
    
    # z_grad deve essere un vettore lungo np.size(points,0)
    grid_z1_x = interpolate.griddata(points, np.reshape(z_grad[0], np.size(XX)), (xyz[...,0], xyz[...,1]), method='linear')
    grid_z1_y = interpolate.griddata(points, np.reshape(z_grad[1], np.size(XX)), (xyz[...,0], xyz[...,1]), method='linear')

    #print(np.size(grid_z1_x))
    #print(np.size(points_interp))
    #sys.exit()

    #takes the absolute max val between x and y direction
    slope      = np.sqrt((grid_z1_x)**2+(grid_z1_y)**2)
    slope_max  = max( np.amax(slope), 1.e-10 )

    # invert slope value for gmsh .pos file
    # taking care of not dividing by zero
    f = 0.9*(  1.0 - (slope/slope_max) ) + 0.1
    scale_factor  = ((np.amax(XX)-np.amin(XX))/(np.size(Lon)-1) + (np.amax(YY)-np.amin(YY))/(np.size(Lat)-1))/2.

    f *= scale_factor *10.
        
    return f
    
def compute_interpolation_error(nodes, triangles, f, z, XX, YY):
    uvw, weights = gmsh.model.mesh.getIntegrationPoints(2, "Gauss2")
    jac, det, pt = gmsh.model.mesh.getJacobians(2, uvw)
    numcomp, sf, _ = gmsh.model.mesh.getBasisFunctions(2, uvw, "Lagrange")
    sf = sf.reshape((weights.shape[0], -1))
    qx = pt.reshape((triangles.shape[0], -1, 3))
    det = np.abs(det.reshape((triangles.shape[0], -1)))
    f_vert = f(nodes, z, XX, YY)
    f_fem = np.dot(f_vert[triangles], sf)
    
    #print(np.shape(nodes))
    #print(np.shape(qx))
    err_tri = np.sum((f_fem - f(qx, z, XX, YY))**2 * det * weights, 1)
    
    #print( f(qx, z, XX, YY))
    #sys.exit()
    return f_vert, np.sqrt(err_tri)
    
def compute_size_field(nodes, triangles, err, N, kk, minres):
    x = nodes[triangles]
    a = 2.
    d = 2.
    fact = (a**((2. + a) /
                (1. + a)) + a**(1. / (1. + a))) * np.sum(err**(2. / (1. + a)))
    ri = err**(2. /
               (2. *
                (1 + a))) * a**(1. /
                                (d *
                                 (1. + a))) * ((1. + a) * N / fact)**(1. / d)
                                 
    cellsize = triangle_max_edge(x) / ri
    mincell = np.amin(cellsize)
    maxcell = np.amax(cellsize)
    if maxcell != mincell:
        Y = minres + (minres * (kk - 1)) / (maxcell - mincell) * (cellsize - mincell)
    else:
        print("maxcell is equal to mincell stop in compute_size_field")
        sys.exit()
        
    return Y

def GetId(LatOrLonDEM, LatOrLonExtent, LatLon_dem_res):
    id = int(np.rint((LatOrLonExtent - LatOrLonDEM)/LatLon_dem_res))
    return id

class Mesh:
    def __init__(self):
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))
        vmap = dict({j: i for i, j in enumerate(self.vtags)})
        self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))



def modifyMeshFile(file, output_file, idFace, a=1):

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


## Number of partition onto which the map will be subdivided: to coincide with an integer submultiple of the images number
Npart = "3"

## Longitude desired resolution on the selected portion of orography
# deg
lon_res = 0.083333333333
## Latitude desired resolution on the selected portion of orography
# deg
lat_res = 0.083333333333


## Longitude coordinates of the desired slice of the orography  [-180, 180)
lon_crop_extent = [-185, -175.0] #[-13.5, 50.0] #[-190, -170.0] # [-190, -170] # thanks to Python that admits negative ids
## Latitude coordinates of the desired slice of the orography [-90, 90]
lat_crop_extent = [0, 10] #[27.7, 64.6] #[-10, 10]


###################################









# Number of elements
#Sphere2Cube(a, lon, lat, idFace)
#mesh_resolution = EARTH_R * np.amin([ (lon_crop_extent[-1] - lon_crop_extent[0]),  (lat_crop_extent[-1] - lat_crop_extent[0])])
#mesh_resolution /= NN

## read orography file
DEM = netCDF4.Dataset('GTOPO_DEM_30s.nc','r')
lon_dem =  DEM.variables['lon'][:]
lat_dem =  DEM.variables['lat'][:]
z_dem   =  DEM.variables['z'][:][:]

## Earth radius in meters
EARTH_R = 6.37122 * 10**6
## half-length of cube edge
cubeRadius = EARTH_R/np.sqrt(3)



# intial data resolution (0.0083333333333 deg)
lon_dem_res = (lon_dem[-1]-lon_dem[0])/ np.size(lon_dem)
lat_dem_res = (lat_dem[-1]-lat_dem[0])/ np.size(lat_dem)

#print(lat_dem_res, lon_dem_res)



# compute matrix index of the slice
lon_id = [GetId(lon_dem[ 0 ], lon_crop_extent[ 0 ], lon_dem_res), GetId(lon_dem[ 0 ], lon_crop_extent[ 1 ], lon_dem_res)]
lat_id = [GetId(lat_dem[ 0 ], lat_crop_extent[ 0 ], lat_dem_res), GetId(lat_dem[ 0 ], lat_crop_extent[ 1 ], lat_dem_res)]


#print(lat_id,lon_id)


# load desired slices
print("Loading data...")
lon_slice = lon_dem[range(lon_id[0], lon_id[1]+1)]
lat_slice = lat_dem[range(lat_id[0], lat_id[1]+1)]
z_slice   = z_dem[  lat_id[0]: lat_id[1]+1, \
                    range(lon_id[0], lon_id[1]+1)]
print("Data loaded")

DEM.close()




## Upscale lon_slice, lat_slice and z_slice
lat_slice_up = np.linspace( lat_crop_extent[0], lat_crop_extent[1], int(np.rint((lat_crop_extent[1]-lat_crop_extent[0])/lat_res))+1 )
lon_slice_up = np.linspace( lon_crop_extent[0], lon_crop_extent[1], int(np.rint((lon_crop_extent[1]-lon_crop_extent[0])/lon_res))+1 )


lon_func_up = np.linspace( lon_slice[0]-360, lon_slice[-1], int(np.rint((lon_slice[-1]-lon_slice[0]+360)/lon_dem_res))+1 )
if lon_func_up[0] < -360:
    lon_func_up = np.linspace( lon_slice[0], lon_slice[-1], int(np.rint((lon_slice[-1]-lon_slice[0])/lon_dem_res))+1 )
func_up = interpolate.interp2d(lon_func_up, lat_slice, z_slice, kind='cubic')
z_slice_up = func_up(lon_slice_up, lat_slice_up)

for ii in range(len(lon_slice_up)):
    if lon_slice_up[ii] < -180:
        lon_slice_up[ii] += 360
    
    



# divide z, lon_slice, lat_slice in at max 6 slices
# following paper nair_et_al_2005

# list if lists
lon_slice_full=[[] for _ in range(6)]
lat_slice_full=[[] for _ in range(6)]
z_slice_full  =[[] for _ in range(6)]

isFaceP1=False
isFaceP2=False
isFaceP3=False
isFaceP4=False
isFaceP5=False
isFaceP6=False


for ii in np.arange(0, len(lon_slice), 20):
    #print(len(lon_slice))
    ll = lon_slice[ ii ]
    for jj in np.arange(0, len(lat_slice), 20):
        th = lat_slice[ jj ]
        
        if ll >= -45 and ll <= 45 and th >= -45 and th <= 45: # P1
            isFaceP1=True
            
        if ll >= 45 and ll <= 135 and th >= -45 and th <= 45: # P2
            isFaceP2=True
            
        if (ll >= 135 and th >= -45 and th <= 45) or (ll <= -135 and th >= -45 and th <= 45): # P3
            isFaceP3=True
        
        if ll >= -135 and ll <= -45 and th >= -45 and th <= 45: # P4
            isFaceP4=True
            
        if th >= 45: # P5
            isFaceP5=True
            
        if th <= -45: # P6
            isFaceP6=True
            

print("isFaceP1 ", isFaceP1)
print("isFaceP2 ", isFaceP2)
print("isFaceP3 ", isFaceP3)
print("isFaceP4 ", isFaceP4)
print("isFaceP5 ", isFaceP5)
print("isFaceP6 ", isFaceP6)

if isFaceP1:
    for ll in lon_slice_up:
        if ll >= -45 and ll <= 45:
            lon_slice_full[ 0 ].append( ll )
    
    lon_slice_full[ 0 ] = np.array(lon_slice_full[ 0 ])
    
    for th in lat_slice_up:
        if th >= -45 and th <= 45:
            lat_slice_full[ 0 ].append( th )
            
    lat_slice_full[ 0 ] = np.array(lat_slice_full[ 0 ])
            
if isFaceP2:
    for ll in lon_slice_up:
        if ll >= 45 and ll <= 135:
            lon_slice_full[ 1 ].append( ll )
        
    lon_slice_full[ 1 ] = np.array(lon_slice_full[ 1 ])
        
    for th in lat_slice_up:
        if th >= -45 and th <= 45:
            lat_slice_full[ 1 ].append( th )
            
    lat_slice_full[ 1 ] = np.array(lat_slice_full[ 1 ])
            
if isFaceP3:
    for ll in lon_slice_up:
        if ll >= 135 or ll <= -135:
            lon_slice_full[ 2 ].append( ll )
            
    lon_slice_full[ 2 ] = np.array(lon_slice_full[ 2 ])
    
    for th in lat_slice_up:
        if th >= -45 and th <= 45:
            lat_slice_full[ 2 ].append( th )
    
    lat_slice_full[ 2 ] = np.array(lat_slice_full[ 2 ])

if isFaceP4:
    for ll in lon_slice_up:
        if ll >= -135 and ll <= -45:
            lon_slice_full[ 3 ].append( ll )

    lon_slice_full[ 3 ] = np.array(lon_slice_full[ 3 ])

    for th in lat_slice_up:
        if th >= -45 and th <= 45:
            lat_slice_full[ 3 ].append( th )
            
    lat_slice_full[ 3 ] = np.array(lat_slice_full[ 3 ])

if isFaceP5:
    for ll in lon_slice_up:
        lon_slice_full[ 4 ].append( ll )

    lon_slice_full[ 4 ] = np.array(lon_slice_full[ 4 ])

    for th in lat_slice_up:
        if th >= 45: # P5:
            lat_slice_full[ 4 ].append( th )
            
    lat_slice_full[ 4 ] = np.array(lat_slice_full[ 4 ])
            
if isFaceP6:
    for ll in lon_slice_up:
        lon_slice_full[ 5 ].append( ll )

    lon_slice_full[ 5 ] = np.array(lon_slice_full[ 5 ])

    for th in lat_slice_up:
        if th <= -45: # P6:
            lat_slice_full[ 5 ].append( th )
            
    lat_slice_full[ 5 ] = np.array(lat_slice_full[ 5 ])
        




for ii in range(len(lon_slice_full)):
    Lon = lon_slice_full[ ii ]
    Lat = lat_slice_full[ ii ]
    if len(Lon) != 0:
        lat_1=GetId(lat_slice_up[ 0 ], Lat[ 0 ],  lat_res)
        lat_2=GetId(lat_slice_up[ 0 ], Lat[ -1 ], lat_res)
        lon_1=GetId(lon_slice_up[ 0 ], Lon[ 0 ],  lon_res)
        lon_2=GetId(lon_slice_up[ 0 ], Lon[ -1 ], lon_res)
        
        if Lon[ 0 ] > Lon[ -1 ]:
            lon_2=GetId(lon_slice_up[ 0 ], Lon[ -1 ]+360, lon_res)
        
        z_slice_full[ ii ] = z_slice_up[lat_1:lat_2+1, range(lon_1,lon_2+1)]

        
        LON, LAT = np.meshgrid( np.radians(Lon), np.radians(Lat) )
        [XX,YY] = Sphere2Cube(LON, LAT, ii)
        
        
        
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.model.add("square")
        index = 1
        for j in range(0, np.size(XX,1)):
            i = 0
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j], 0, tag=index)
            if j > 0:
                gmsh.model.geo.addLine(index, index-1, index-1)
            index += 1
        
        for i in range(1, np.size(XX,0)):
            j = np.size(XX,1)-1
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j], 0, tag=index)
            gmsh.model.geo.addLine(index, index-1, index-1)
            index += 1
        
        for j in range(np.size(XX,1)-2, 0, -1):
            i = np.size(XX,0)-1
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j], 0, tag=index)
            gmsh.model.geo.addLine(index, index-1, index-1)
            index += 1
        
        for i in range(np.size(XX,0)-1, 0, -1):
            j = 0
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j], 0, tag=index)
            gmsh.model.geo.addLine(index, index-1, index-1)
            index += 1
        
        gmsh.model.geo.addLine(1, index-1, index-1) # close the 'square'
        
        #gmsh.write("model.brep")
        
        gmsh.model.geo.addCurveLoop(np.linspace(1,index-1,index-1), 1)
        gmsh.model.geo.addPlaneSurface([1], 6)
        gmsh.model.geo.synchronize()
        #pnts = gmsh.model.getBoundary([6])
        #print(pnts)
        #gmsh.model.mesh.setSize(pnts, 0.1)
        gmsh.model.mesh.generate(2)
        mesh = Mesh()
        #gmsh.write("mesh.msh")
        

        f_nod, err_ele = compute_interpolation_error(mesh.vxyz, mesh.triangles,
                                             grad_func, z_slice_full[ ii ], XX, YY)
        
        kk = 50
        minres = ((np.amax(XX)-np.amin(XX))/(np.size(Lon)-1) + (np.amax(YY)-np.amin(YY))/(np.size(Lat)-1))/2.
        minres /= 10
        
        f_view = gmsh.view.add("nodal function")
        gmsh.view.addModelData(f_view, 0, "square", "NodeData", mesh.vtags,
                       f_nod[:, None])
        gmsh.view.write(f_view, "f.pos")
        err_view = gmsh.view.add("element-wise error")
        gmsh.view.addModelData(err_view, 0, "square", "ElementData",
                       mesh.triangles_tags, err_ele[:, None])
        gmsh.view.write(err_view, "err.pos")
        sf_ele = compute_size_field(mesh.vxyz, mesh.triangles, err_ele, 1e3, kk, minres)
        sf_view = gmsh.view.add("mesh size field")
        gmsh.view.addModelData(sf_view, 0, "square", "ElementData",
                       mesh.triangles_tags, sf_ele[:, None])
        gmsh.view.write(sf_view, "sf.pos")
                       
                
        gmsh.model.add("square2")
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        toll = 1e-6
        index = 1
        pList1 = []
        pList2 = []
        pList3 = []
        pList4 = []
        for j in range(0, np.size(XX,1)):
            i = 0
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j]+toll, 0, tag=index)
            pList1.append(index)
            index += 1
        
        pList2.append(index-1)
        for i in range(1, np.size(XX,0)):
            j = np.size(XX,1)-1
            gmsh.model.geo.addPoint(XX[i,j]-toll, YY[i,j], 0, tag=index)
            pList2.append(index)
            index += 1
        
        pList3.append(index-1)
        for j in range(np.size(XX,1)-2, 0, -1):
            i = np.size(XX,0)-1
            gmsh.model.geo.addPoint(XX[i,j], YY[i,j]-toll, 0, tag=index)
            pList3.append(index)
            index += 1
        pList3.append(index)
                
        for i in range(np.size(XX,0)-1, 0, -1):
            j = 0
            gmsh.model.geo.addPoint(XX[i,j]+toll, YY[i,j], 0, tag=index)
            pList4.append(index)
            index += 1
        pList4.append(1)
        
        
        gmsh.model.geo.addSpline(pList1, tag=1)
        gmsh.model.geo.addSpline(pList2, tag=2)
        gmsh.model.geo.addSpline(pList3, tag=3)
        gmsh.model.geo.addSpline(pList4, tag=4)
        
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        #gmsh.model.addPhysicalGroup(2, [1], 1)
                       
        #gmsh.model.add("square2")
        #dx = abs(XX[0,np.size(XX,1)-1] - XX[0,0])
        #dy = abs(YY[np.size(XX,0)-1,0] - YY[0,0])
        #toll = 1e-5
        #gmsh.model.geo.addPoint(XX[0,0]+toll, YY[0,0]+toll, 0, tag=1)
        #gmsh.model.geo.addPoint(XX[0,0]+dx-toll, YY[0,0]+toll,  0, tag=2)
        #gmsh.model.geo.addPoint(XX[0,0]+dx-toll, YY[0,0]+dy-toll, 0, tag=3)
        #gmsh.model.geo.addPoint(XX[0,0]+toll, YY[0,0]+dy-toll, 0, tag=4)
        #gmsh.model.geo.addLine(1, 2, 1)
        #gmsh.model.geo.addLine(2, 3, 2)
        #gmsh.model.geo.addLine(3, 4, 3)
        #gmsh.model.geo.addLine(4, 1, 4)
        #gmsh.model.geo.addCurveLoop([4, 1, 2, 3], 1)
        #gmsh.model.geo.addPlaneSurface([1], 1)
        #gmsh.model.addPhysicalGroup(2, [1], 1)
        
        
        gmsh.model.geo.synchronize()
        bg_field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", sf_view)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Format", 30)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.option.setNumber("Mesh.SaveElementTagType", 3)
        
        
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh_"+str(ii)+".mesh")
        gmsh.finalize()
        
        
        modifyMeshFile("mesh_"+str(ii)+".mesh", "mesh_mod_"+str(ii)+".mesh", ii)
        #writeDatFileQuad("mesh_mod_"+str(ii)+".mesh", "mesh_"+str(ii)+".dat")
        

# now merge the multiple dat files

# prima processare tutti i .mesh per vedere se ci sono punti 'uguali', a meno della maledetta tolleranza, nei lati di bordo, poi usare merge di .mesh, una volta che si ha un unico .mesh file convertire in .dat

# prima fare una lista di boundary nodes per ogni face


isFace_vect = [isFaceP1, isFaceP2, isFaceP3, isFaceP4, isFaceP5, isFaceP6]
for ii in range(0,6):
    isFace = isFace_vect[ ii ]
    if isFace:
        file_o = open("mesh2.mesh","r")
        





os.remove("*.mesh")




sys.exit()





## create dat files from mesh files


#
file_o = open("mesh2.mesh","r")
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

    if line.strip() == "Hexahedra":
        nbHex = listOfLines[i+1]
        file.write(nbHex)
        for j in range(int(nbHex)):
            k = j+i+1+1

            myarray = listOfLines[k].split()
            file.write(myarray[8] + " " + myarray[0] + " " + myarray[1] + " " + myarray[2] + " " + myarray[3] + " " + myarray[4] + " "+ myarray[5] + " " + myarray[6] + " " + myarray[7] + "\n")
            #file.write(listOfLines[k])

file_o.close()
file.close()



os.remove(fy_geo )
os.remove(fy_pos )
os.remove(fy_mesh)




sys.exit()


