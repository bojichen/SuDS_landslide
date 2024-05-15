#!/usr/bin/env python
# coding: utf-8

# # Import the required libraries

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import flopy
import pandas as pd
from osgeo import gdal
from pyswmm import Simulation, Subcatchments, SystemStats, LidGroups
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')


# # Creation of model object and application of MODFLOW NWT

# In[2]:


modelname = 'Mukilteo'
work_ws = 'SWMM_Modflow_01'
ml = flopy.modflow.Modflow(modelname, exe_name="/home/bojichen/.local/share/flopy/bin/mfnwt", version='mfnwt', model_ws=work_ws)

#Defininition of MODFLOW NWT Solver
# maxiterout is the maximum number of iterations for the outer (nonlinear) loop
# linmeth is the linearization method. 1 is Picard, 2 is modified Picard, 3 is Newton
# headtol is the head change criterion for convergence
nwt = flopy.modflow.ModflowNwt(ml, maxiterout=200, linmeth=2, headtol=0.01)


# In[3]:


# read the DEM raster and mask
demPath = "./GIS/dem3m_bound.tif"
crPath = "./GIS/mask_final.tif"


# In[4]:


demDs =gdal.Open(demPath)  # type: gdal.Dataset
crDs = gdal.Open(crPath)
geot = crDs.GetGeoTransform() #Xmin, deltax, ?, ymax, ?, delta y


# In[5]:


demData = demDs.GetRasterBand(1).ReadAsArray()   # demData is a 2D array
crData = crDs.GetRasterBand(1).ReadAsArray()


# In[6]:


# shape the size of DEM as crData
demData = demData[:1270,:1030]
demData[demData<0] = 0 


# In[7]:


Layer1 = demData - 2
Layer2 = demData - 4
Layer3 = (Layer2 + 10) * 0.8 - 10
Layer4 = (Layer2 + 10) * 0.5 - 10
Layer5 = -10


# In[8]:


#Boundaries for Dis = Create discretization object, spatial/temporal discretization
ztop = demData
zbot = [Layer1, Layer2, Layer3, Layer4, Layer5]
nlay = 5
nrow = demData.shape[0]
ncol = demData.shape[1]
delr = geot[1]
delc = abs(geot[5])


# In[9]:


# parameters for time discretization
nper = 1   # number of stress periods
perlen = 1  # length of stress period
nstp = 1  # number of time steps
steady = False


# In[10]:


# Time discretization
dis = flopy.modflow.ModflowDis(ml, nlay,nrow,ncol,delr=delr,delc=delc,top=ztop,botm=zbot,
                               nper=nper, perlen=perlen, nstp=nstp, steady=steady)


# In[11]:


# UPW package
# hk = [1E-4, 1E-5, 1E-7] unit is m/s convert to m/day
hk = [x * 86400 for x in [1.5E-5, 1.5E-5, 1.5E-5, 5E-6, 5E-7]] 
# laytyp = [1,1,0]  # 1 is convertible, 0 is confined, specify the type of layer
upw = flopy.modflow.ModflowUpw(ml, laytyp = [1,1,1,1,0], sy = [0.05, 0.05, 0.05, 0.05, 0.05], hk = hk, ipakcb=53)   # uptream weight package


# In[12]:


# set basic package 
iboundData = np.zeros(demData.shape, dtype=np.int32)
iboundData[crData != 0 ] = 1

fname = os.path.join('SWMM_Modflow', 'Mukilteo.hds')
headfile = flopy.utils.HeadFile(fname)
heads = headfile.get_data()
heads[heads==1.e+30] = np.nan            # fix masked data 
heads[heads==-999.99] = np.nan
strt = heads[0]


# In[13]:
riv = np.zeros((nrow, ncol), dtype=np.float32)
riv[(crData ==333)] = 1
list = []
for i in range(nrow):
    for j in range(ncol):
        if riv[i,j] == 1:
            list.append([0,i,j,ztop[i,j], 864, ztop[i,j]-1])  # [layer, row, column, stage, cond, rbot]
riv_spd = {0: list}
riv = flopy.modflow.ModflowRiv(ml, ipakcb=53, stress_period_data=riv_spd)


# the constant-boundary condition
chd_array = np.zeros(demData.shape, dtype=np.int32)
chd_array[(crData == 222)] = 1
lst = []
for k in range(nlay):
    for i in range(chd_array.shape[0]):
        for q in range(chd_array.shape[1]):
            if chd_array[i,q] == 1:
                elevation = ztop[i,q]
                lst.append([k,i,q,elevation,elevation]) #layer,row,column, starting head, ending head
chd_spd = {0:lst}
chd = flopy.modflow.ModflowChd(ml, stress_period_data=chd_spd)


# In[14]:


# output control packages
spd = {(0,0): ['save head', 'save budget']}
oc = flopy.modflow.ModflowOc(ml, stress_period_data=spd) 


# In[15]:


# evtr = np.zeros((nrow, ncol), dtype=np.float32)
# for i in range(evtr.shape[0]):
    # for j in range(evtr.shape[1]):
        # if crData[i,j] != 0:
            # evtr[i,j] = 1 /365
# evtr_data = {0: evtr}
# evt = flopy.modflow.ModflowEvt(ml,nevtop=1,surf=ztop,evtr=evtr_data, exdp=1, ipakcb=53)


# In[16]:


from pysheds.grid import Grid

grid = Grid.from_raster(r'./GIS/dem3m_bound.tif')
dem = grid.read_raster(r'./GIS/dem3m_bound.tif')

# Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)
    
# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)

dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

acc = grid.accumulation(fdir, dirmap=dirmap)


# In[17]:


# create a drain package
drn_array = np.zeros(demData.shape, dtype=np.int32)
lst = []
for i in range(drn_array.shape[0]):
    for q in range(drn_array.shape[1]):
        if ((ztop[i,q] < 40 or acc[i,q] > 1000) and crData[i,q] != 222 and crData[i,q] != 333 and iboundData[i,q] == 1):
            drn_array[i,q] == 1
            elevation = ztop[i,q]
            lst.append([0,i,q,elevation, hk[0] * 10]) #layer,row,column, starting head, ending head
drn_spd = {0:lst}
drn = flopy.modflow.ModflowDrn(ml, stress_period_data=drn_spd)


# In[18]:


SWMM_path = 'mukilteo_distant_10%.inp'


# In[20]:


# zon = np.repeat(crData[np.newaxis, :, :], nlay, axis=0)   # by adding one more dimension, the shape of zon is (3, 1270, 1030)


# In[21]:


# ZB_TS=[]
HD_TS=[]


# In[22]:


# %%time
# %%capture

proc_bar = tqdm(total=100)

with Simulation(SWMM_path) as sim:
    
    system_routing = SystemStats(sim)
    #Lists of cummulative infiltration to calculate delta infiltratation for every time step: S, WSU and FSU 
    S_list = []
    array = np.arange(1, 47).astype(str)
    S_name_list = ['S' + num for num in array]

    
    for s in S_name_list:
        S_list.append(Subcatchments(sim)[s])
        
    subcatchments = Subcatchments(sim)
    S_area = []
    for i in S_name_list:
        S_area.append(subcatchments[i].area*10000)   # why here multiply 10000? because the unit of area in SWMM is hectare, 1 hectare = 10000 m^2
    
    inf_S_list_1 = np.zeros(len(S_list))
    inf_S_list_2 = np.zeros(len(S_list))
    
    #time counter for daily infiltration agregation and hourly reports
    step_counter = 0
    day_counter = 0
    hourly_counter = 0
    for step in sim:
        step_counter=step_counter+1

        if step_counter == 360: #CHANGE ACCORDING TO DT
            step_counter = 0
            hourly_counter += 1
                            
        #DAILY INFILTRATION ON SUBCATCHMENT:
        
        if hourly_counter==24:
            day_counter=day_counter+1
            proc_bar.update(100/391)

            hourly_counter=0
            
            print(sim.current_time)

            for i in range(len(S_list)):
                #Delta infiltration
                #CHANGE OF UNITS m3/day->m/day:
                nLidUnits = LidGroups(sim)[S_list[i].subcatchmentid]._nLidUnits
                if nLidUnits > 0:
                    Cummulative_infil_Lid = LidGroups(sim)[S_list[i].subcatchmentid][0].water_balance.infiltration * 1E-3
                    Area_conversion = (LidGroups(sim)[S_list[i].subcatchmentid][0].unit_area * LidGroups(sim)[S_list[i].subcatchmentid][0].number)/S_area[i]
                    LID_infil = Cummulative_infil_Lid * Area_conversion
                        
                else:
                    LID_infil = 0

                inf_S_list_2[i]=(S_list[i].statistics["infiltration"]/S_area[i] + LID_infil -inf_S_list_1[i])
                inf_S_list_1[i]=S_list[i].statistics["infiltration"]/S_area[i] + LID_infil
                
            RCH_S = inf_S_list_2

            RCH_S_df = pd.DataFrame({"S":S_name_list, "RCH_S":RCH_S})
            
            rch_array = np.zeros((nrow, ncol))
            for i in range(nrow):
                for j in range(ncol):
                    cell = "S" + str(crData[i, j])
                    if cell in S_name_list:
                        flux = RCH_S_df.loc[RCH_S_df['S'] == cell, 'RCH_S'].values[0]
                        rch_array[i, j] = flux
                    else:
                        rch_array[i, j] = 0
            rch_data = {0:rch_array}

            # print(f"RECHARGE at (558, 230) is {rch_array[558, 230]}")

            rch = flopy.modflow.ModflowRch(ml, nrchop=3, rech=rch_data, ipakcb=53)
            
            bas = flopy.modflow.ModflowBas(ml, ibound=iboundData, strt=strt)
            
            ml.write_input()
            ml.run_model(silent=True)
            
            #Read MODFLOW outputs
            fname = os.path.join('SWMM_Modflow_01', 'Mukilteo.hds')
            headfile = flopy.utils.HeadFile(fname, model=ml)
            heads = headfile.get_data()
            heads[heads==1.e+30] = np.nan            # fix masked data 
            heads[heads==-999.99] = np.nan

            #Strt next loop
            
            strt = heads[0]
            # print(f"Starting head at (558, 230) is {strt[558, 230]}")
            HD_TS.append(strt)
            
            # fname = os.path.join('SWMM_Modflow', 'Mukilteo.cbc')
            # cbb = flopy.utils.CellBudgetFile(fname)
            # zb = flopy.utils.ZoneBudget(cbb, zon)
            # zb_df=zb.get_dataframes()
            # ZB_TS.append(zb_df)
            
    # routing_stats=system_routing.routing_stats
    # runoff_stats=system_routing.runoff_stats

proc_bar.close()


# In[27]:

HD_TS_SUM = []
HD_TS_VAR = []
for i in range(len(HD_TS)):
    HD_TS_SUM.append(np.nansum(HD_TS[i]))
    HD_TS_VAR.append(np.nanvar(HD_TS[i]))
np.save('HD_TS_SUM_Distant_10.npy', HD_TS_SUM)
np.save('HD_TS_VAR_Distant_10.npy', HD_TS_VAR)

max_index = np.nanargmax(HD_TS_SUM)
min_index = np.nanargmin(HD_TS_SUM)

np.save('HD_TS_MAX_Distant_10.npy', HD_TS[max_index])
np.save('HD_TS_MIN_Distant_10.npy', HD_TS[min_index])