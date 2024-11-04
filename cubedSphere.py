import sys
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.spatial import Delaunay
import netCDF4
import matplotlib.pyplot as plt
import gmsh  
import meshio

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
        print("Not recognized cube-face Id, STOP!")
        sys.exit()
        
    return [x, y]
    


def XYZ2LonLat(X, Y, Z, lon_crop_extent):

    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    lat = np.arcsin(Z/r)*180./np.pi
    lon = np.arctan2(Y,X)*180./np.pi
    
    mask = np.logical_and(X==0, Y==0)
    if np.size(lon)==1:
        if mask==True:
            lon = lon_crop_extent[0]
    else:
        lon[mask] = lon_crop_extent[0]
    
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
        print("Not recognized cube-face Id, STOP!")
        sys.exit()
        
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
        print("Not recognized cube-face Id, STOP!")
        sys.exit()
        
        
    return [x, y]
    
def computeGrad_spherical(z_slice, lat_dem_res, lon_dem_res, LON, LAT, R=np.sqrt(3.)):
    d_z_lat_slice = np.gradient(z_slice, lat_dem_res*np.pi/180, axis=0)
    d_z_lon_slice = np.gradient(z_slice, lon_dem_res*np.pi/180, axis=1)
    
    d_z_lat = (1./R)*d_z_lat_slice
    d_z_lon =  1./(R*np.cos(np.radians(LAT)))*d_z_lon_slice

    return [d_z_lat, d_z_lon]

def inverse_metric_tensor(Localx, Localy, a=1):
    R = a*np.sqrt(3)
    r = np.sqrt(Localx**2 + Localy**2 + a**2)
    scale_coeff = R**2/r**4
    g11 =  scale_coeff*(a**2 + Localy**2)
    g12 = -scale_coeff*Localx*Localy
    g21 = g12
    g22 = scale_coeff*(a**2 + Localx**2)
    g_sqrt = R**2*a/r**3
    g_det = (g_sqrt)**2

    gap11 = g22/g_det
    gap12 = -g12/g_det
    gap21 = -g21/g_det
    gap22 = g11/g_det
    ginv = np.array([[gap11, gap12], [gap21, gap22]])
    #g = np.array([[g11, g12], [g21, g22]])
    return [g_sqrt, ginv]

def localBasis2Spherical(x, y, lon_crop_extent, idFace, a=1):
    R = a*np.sqrt(3.)

    if idFace == 0 or idFace == 1 or idFace == 2 or idFace == 3:
        lx = -(a/(a**2+x**2))
        ly = lx*0
        tx = -((R*x*y)/(np.sqrt((R**2*(a**2+x**2))/(a**2+x**2+y**2))*(a**2+x**2+y**2)**(3./2.)))
        ty = np.sqrt((R**2*(a**2+x**2))/(a**2+x**2+y**2))/(R*np.sqrt(a**2+x**2+y**2))
    
    elif idFace == 4:
        lx = y/(x**2+y**2)
        ly = -(x/(x**2+y**2))
        tx = -((a*R*x)/(np.sqrt((R**2*(x**2+y**2))/(a**2+x**2+y**2))*(a**2+x**2+y**2)**(3./2.)))
        ty = -((a*R*y)/(np.sqrt((R**2*(x**2+y**2))/(a**2+x**2+y**2))*(a**2+x**2+y**2)**(3./2.)))
        
    elif idFace == 5:
        lx = -(y/(x**2+y**2))
        ly = x/(x**2+y**2)
        tx = (a*R*x)/(np.sqrt((R**2*(x**2+y**2))/(a**2+x**2+y**2))*(a**2+x**2+y**2)**(3./2.))
        ty = (a*R*y)/(np.sqrt((R**2*(x**2+y**2))/(a**2+x**2+y**2))*(a**2+x**2+y**2)**(3./2.))
         
    else: 
        print("Not recognized cube-face Id, STOP!")
        sys.exit()

    [X,Y,Z] = Local2Global(x, y, idFace)
    [t, l] = XYZ2LonLat(X, Y, Z, lon_crop_extent)

    A = np.array([[R*lx*np.cos(t), R*tx], [R*ly*np.cos(t), R*ty]])
    return [A,lx,ly,tx,ty]

def computeParDer(xx_1, xx_2, xx_3, z_node_triangle, iCubeFace):
    z0 = z_node_triangle[0]
    z1 = z_node_triangle[1]
    z2 = z_node_triangle[2]

    x0 = xx_1[0]
    x1 = xx_2[0]
    x2 = xx_3[0]

    y0 = xx_1[1]
    y1 = xx_2[1]
    y2 = xx_3[1]

    a2 = (x1*z0-x2*z0-x0*z1+x2*z1+x0*z2-x1*z2)/(x1*y0-x2*y0-x0*y1+x2*y1+x0*y2-x1*y2)
    a1 = (y2*(-z0+z1)+y1*(z0-z2)+y0*(-z1+z2))/(x2*(y0-y1)+x0*(y1-y2)+x1*(-y0+y2))

    return np.array([a1, a2])



    
def compute_size_field(nodes, triangles, oro_interp, d_z_lat_interp, d_z_lon_interp, lon_crop_extent, delta_min=1e-4, delta_max=.05, tau=.1, a=1):
    
    xyz = nodes[triangles]
    
    lat_interp_node, lon_interp_node = XYZ2LonLat(nodes[:,0], nodes[:,1], nodes[:,2], lon_crop_extent)
    
    z_node = oro_interp((lat_interp_node, lon_interp_node), method='linear')
    
    
    
#    plt.plot(lon_interp_node, lat_interp_node, 'o')
#    plt.axis('equal')
#    plt.show()
    
#    plt.triplot(x_local, y_local, triangles)
#    plt.tricontourf(x_local, y_local, triangles, z_node)

#    plt.tricontourf(lon_interp_node, lat_interp_node, triangles, z_node)
#    plt.axis('equal')
#    plt.colorbar()
#    plt.show()
    
    
    z_node = z_node[triangles]
    
    xyz_middle = (xyz[:,0,:] + xyz[:,1,:] + xyz[:,2,:])/3.
    lat_interp_middle, lon_interp_middle = XYZ2LonLat(xyz_middle[:,0], xyz_middle[:,1], xyz_middle[:,2], lon_crop_extent)
    d_z_lat_middle = d_z_lat_interp((lat_interp_middle, lon_interp_middle), method='nearest')
    d_z_lon_middle = d_z_lon_interp((lat_interp_middle, lon_interp_middle), method='nearest')
    

    
    N_elements = xyz.shape[0]
    sf = np.zeros(len(nodes))+1e4
    
    iCubeFace = -1
    for iTri in range(xyz.shape[0]):
        xyz_triangle = xyz[iTri]
        z_node_triangle = z_node[iTri]

        xx_1 = xyz_triangle[0,:]
        xx_2 = xyz_triangle[1,:]
        xx_3 = xyz_triangle[2,:]
            
        
        if (np.allclose(xx_1[0],a) and np.allclose(xx_2[0],a) and np.allclose(xx_3[0],a)):
            iCubeFace = 0
        elif (np.allclose(xx_1[1],a) and np.allclose(xx_2[1],a) and np.allclose(xx_3[1],a)):
            iCubeFace = 1
        elif (np.allclose(xx_1[0],-a) and np.allclose(xx_2[0],-a) and np.allclose(xx_3[0],-a)):
            iCubeFace = 2
        elif (np.allclose(xx_1[1],-a) and np.allclose(xx_2[1],-a) and np.allclose(xx_3[1],-a)):
            iCubeFace = 3
        elif (np.allclose(xx_1[2],a) and np.allclose(xx_2[2],a) and np.allclose(xx_3[2],a)):
            iCubeFace = 4
        elif (np.allclose(xx_1[2],-a) and np.allclose(xx_2[2],-a) and np.allclose(xx_3[2],-a)):
            iCubeFace = 5
        else:
            print("Not recognized cube-face Id, STOP!")
            sys.exit()

        measure_tri = .5*np.linalg.norm(np.cross(xx_2 - xx_1, xx_3 - xx_1))

        [Localx, Localy] = Global2Local(xyz_middle[0,0], xyz_middle[0,1], xyz_middle[0,2], iCubeFace)
        [g_sqrt, ginv] = inverse_metric_tensor(Localx, Localy)

        [A,lx,ly,tx,ty] = localBasis2Spherical(Localx, Localy, lon_crop_extent, iCubeFace)
        xxl_1 = Global2Local(xx_1[0], xx_1[1], xx_1[2], iCubeFace)
        xxl_2 = Global2Local(xx_2[0], xx_2[1], xx_2[2], iCubeFace)
        xxl_3 = Global2Local(xx_3[0], xx_3[1], xx_3[2], iCubeFace)
        vec_par_der_xy = computeParDer(xxl_1, xxl_2, xxl_3, z_node_triangle, iCubeFace)
                             
        left_contr = ginv@(np.array([d_z_lon_middle[iTri]*lx + d_z_lat_middle[iTri]*tx, d_z_lon_middle[iTri]*ly + d_z_lat_middle[iTri]*ty]))@A
        right_contr = ginv@vec_par_der_xy@A

        eta_k = np.sqrt(np.linalg.norm(left_contr - right_contr)**2*measure_tri*g_sqrt)

        candidate = tau*np.sqrt(measure_tri/N_elements/.5)/(eta_k+1.e-8)
        
        for iPoi in range(3):
            sf[triangles[iTri][iPoi]] = np.minimum(sf[triangles[iTri][iPoi]], np.maximum(np.minimum(candidate, delta_max), delta_min)) #np.maximum(np.minimum(candidate, starting_mesh_size*4), starting_mesh_size/2) #np.maximum(np.minimum(candidate, starting_mesh_size*2), starting_mesh_size/2)
    
    for i in range(len(sf)):
        if sf[i]==1e4:
            Poi = nodes[i]
            d = np.sum(np.abs(Poi-nodes)**2,axis=-1)**(1./2.)
            b = np.ma.MaskedArray(d, (d==0)+(sf==1e4))
            index_c = np.ma.argmin(b)
            sf[i] = sf[index_c]

    return sf 
    

def GetId(LatOrLonDEM, LatOrLonExtent, LatLon_dem_res):
    id = int(((LatOrLonExtent - LatOrLonDEM)/LatLon_dem_res))
    return id

class Mesh:
    def __init__(self):
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        
        self.vxyz = vxyz.reshape((-1, 3))
        vmap = dict({j: i for i, j in enumerate(self.vtags)})
        self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))


            

###################################
## Inputs
## Longitude coordinates of the desired slice of the orography  [-180, 180)
lon_crop_extent = [30, 50]
## Latitude coordinates of the desired slice of the orography [-90, 90]
lat_crop_extent = [30, 40]

earth_radius = 6.371e6

MeshAlgorithm = 5 # Gmsh meshing algorithm, prefer to use MeshAdapt

# baseline mesh size
starting_mesh_size = 1e-3

delta_min = 1e-3
delta_max = .01 # coarsest mesh resolution
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
z_slice   = z_slice.data/earth_radius*np.sqrt(3)

if (lon_slice[0] > lon_slice[-1]): # non ascending order, pb with RegularGridInterpolator
    lon_slice[lon_slice >= lon_slice[0]] -= 360.

LON, LAT = np.meshgrid(lon_slice, lat_slice)

plt.contourf(LON, LAT, z_slice)
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()


# structured grid, faster than stadard scipy interpolation
z_mesh_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), z_slice, fill_value=None, bounds_error=False, method='linear')
d_z_lat, d_z_lon = computeGrad_spherical(z_slice, lat_dem_res, lon_dem_res, LON, LAT)
d_z_lat_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), d_z_lat, fill_value=None, bounds_error=False, method='nearest')
d_z_lon_fem = interpolate.RegularGridInterpolator((lat_slice, lon_slice), d_z_lon, fill_value=None, bounds_error=False, method='nearest')



plt.contourf(LON, LAT, np.arctan(np.sqrt(d_z_lat**2 + d_z_lon**2))*180/np.pi)
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar()
plt.show()





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
mask_crop_boundary = mask_crop_boundary.astype(int) - ndimage.binary_erosion(mask_crop_boundary.astype(int))
mask_crop_boundary = mask_crop_boundary.astype(bool)




mask_crop_boundary_mod = mask_crop_boundary.astype(int)
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

#mask_crop_boundary_mod[np.abs(LON_mod-45)<toll] += 1


mask_crop_boundary_mod[0 , 0] += 1
mask_crop_boundary_mod[-1, 0] += 1
mask_crop_boundary_mod[0 ,-1] += 1
mask_crop_boundary_mod[-1,-1] += 1

plt.plot(LON_mod[mask_crop_boundary_mod>0], LAT_mod[mask_crop_boundary_mod>0], 'o')
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()


mask_crop_boundary_mod = mask_crop_boundary_mod>1


plt.plot(LON_mod[mask_crop_boundary_mod], LAT_mod[mask_crop_boundary_mod], 'o')
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()



    



mask = np.logical_and(np.abs(LAT)<np.arctan(np.cos(LON*np.pi/180.))*180./np.pi, np.cos(LON*np.pi/180)>1./np.sqrt(2.))
if (LAT[mask].size!=0 and LON[mask].size!=0):
    isFaceP1 = True
    
    print("isFaceP1 ", isFaceP1)
    
    bool_val = np.logical_not(np.logical_and(np.abs(LAT_mod)<np.arctan(np.cos(LON_mod*np.pi/180.))*180./np.pi+toll, np.cos(LON_mod*np.pi/180)>1./np.sqrt(2.)-toll))
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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
    mask_crop_boundary_p = mask_crop_boundary_mod.astype(bool)
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


for i in range(6):
    if isFace_vect[ i ]:
        plt.triplot(tri[i].points[:,0], tri[i].points[:,1], tri[i].simplices)
        plt.plot(tri[i].points[:,0], tri[i].points[:,1], 'o')

plt.xlabel('longitude')
plt.ylabel('latitude')
plt.axis('equal')
plt.show()


# build sf_view
gmsh.initialize()
gmsh.option.setNumber("Mesh.RecombineAll", 0)
gmsh.option.setNumber("Mesh.Algorithm", MeshAlgorithm)
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
    

        eList_surf = []
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
                    
                    p1_vect = [p1[0], p2[0]]
                    p2_vect = [p1[1], p2[1]]
                    
                    if ((p1[0] == p2[0]) or (p1[1] == p2[1])):
                        p1_vect = np.linspace(p1[0], p2[0])
                        p2_vect = np.linspace(p1[1], p2[1])
                        
                    XX, YY = Sphere2Cube(p1_vect, p2_vect, lat_dem_res, lon_dem_res, ii)
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
                    
                    
                    if (np.size(pList)>2):
                        gmsh.model.geo.addSpline(pList, tag=current_e)
                    else:
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
            
            
            
            is_entered = False
            for ee in eList:
                if (ee in eList_surf):
                    is_entered = True
                    for ee_int in eList:
                        if (ee_int not in eList_surf):
                            eList_surf.append(-ee_int)
                        else:
                            eList_surf.remove(ee_int)
                        
                    break
                    
                if (-ee in eList_surf):
                    is_entered = True
                    for ee_int in eList:
                        if (-ee_int not in eList_surf):
                            eList_surf.append(ee_int)
                        else:
                            eList_surf.remove(-ee_int)
                            
                    break

            
            if (not is_entered):
                for ee_int in eList:
                    eList_surf.append(ee_int)
                    
            
        
        for ee in eList_surf:
            if (np.abs(ee) not in eList_full):
                eList_full.append(ee)
            
        gmsh.model.geo.addCurveLoop(eList_surf, iLoop)
        gmsh.model.geo.addPlaneSurface([iLoop], iSurf)
        iLoop += 1
        iSurf += 1
            
        
        

gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, eList_full, 1)
gmsh.model.mesh.generate(2)
gmsh.write("mesh_starting.msh")
            
mesh = Mesh()
            
print("mesh elements, ", mesh.triangles.shape[0])

# boundary nodes       
nodes_coords = np.reshape(gmsh.model.mesh.getNodesForPhysicalGroup(1,1)[1],(-1,3)) 
nodes_ids    = gmsh.model.mesh.getNodesForPhysicalGroup(1,1)[0]




sf_ele = compute_size_field(mesh.vxyz, mesh.triangles, z_mesh_fem, d_z_lat_fem, d_z_lon_fem, lon_crop_extent, delta_min, delta_max)
sf_node = nodes_ids/nodes_ids*starting_mesh_size

#sf_tot = np.concatenate((sf_ele, sf_node))
#tags_tot = np.concatenate((mesh.vtags, nodes_ids))
sf_tot = sf_ele
tags_tot = mesh.vtags
            
sf_view = gmsh.view.add("mesh size field") # ElementNodeData
#gmsh.view.addModelData(sf_view, 0, current_model, "ElementNodeData", mesh.triangles_tags, sf_ele[:, None])
#gmsh.view.addModelData(sf_view, 0, current_model, "ElementData", mesh.triangles_tags, sf_ele[:, None])
gmsh.view.addModelData(sf_view, 0, current_model, "NodeData", tags_tot, sf_tot[:,None])
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
    

        eList_surf = []
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
                    
                    p1_vect = [p1[0], p2[0]]
                    p2_vect = [p1[1], p2[1]]
                    
                    if ((p1[0] == p2[0]) or (p1[1] == p2[1])):
                        p1_vect = np.linspace(p1[0], p2[0])
                        p2_vect = np.linspace(p1[1], p2[1])
                        
                    XX, YY = Sphere2Cube(p1_vect, p2_vect, lat_dem_res, lon_dem_res, ii)
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
            
                    if (np.size(pList)>2):
                        gmsh.model.geo.addSpline(pList, tag=current_e)
                    else:
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
                    
            
            is_entered = False
            for ee in eList:
                if (ee in eList_surf):
                    is_entered = True
                    for ee_int in eList:
                        if (ee_int not in eList_surf):
                            eList_surf.append(-ee_int)
                        else:
                            eList_surf.remove(ee_int)
                        
                    break
                    
                if (-ee in eList_surf):
                    is_entered = True
                    for ee_int in eList:
                        if (-ee_int not in eList_surf):
                            eList_surf.append(ee_int)
                        else:
                            eList_surf.remove(-ee_int)
                            
                    break

            
            if (not is_entered):
                for ee_int in eList:
                    eList_surf.append(ee_int)

                
        gmsh.model.geo.addCurveLoop(eList_surf, iLoop)
        gmsh.model.geo.addPlaneSurface([iLoop], iSurf)
        iLoop += 1
        iSurf += 1
            
            
gmsh.model.geo.synchronize()

bg_field = gmsh.model.mesh.field.add("PostView")
gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", sf_view)
gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
gmsh.model.mesh.generate(2)
gmsh.write("mesh_final_cube.mesh")



mesh_io = meshio.read("mesh_final_cube.mesh")

lat, lon = XYZ2LonLat(mesh_io.points[:,0], mesh_io.points[:,1], mesh_io.points[:,2], lon_crop_extent)
mesh_io.points[:,0] = lon
mesh_io.points[:,1] = lat
mesh_io.points[:,2] = np.sqrt(3.)


meshio.write("mesh_final.mesh", mesh_io)


gmsh.finalize()











