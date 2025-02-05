# aggregating E-OBS temperature data to NUTS boundaries
#### Christopher Callahan
#### christophercallahan@stanford.edu


import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import xagg as xa
import sys
import os
import geopandas as gp
from tqdm import tqdm
import datetime

# data locations
folder = "/your/path/to/project/folder/"
loc_tx = "/path/to/E-OBS/dailymaxtemp/"
loc_tn = "/path/to/E-OBS/dailymintemp/"
loc_pr =  "/path/to/E-OBS/dailyprecip/"
loc_shp = folder+"Data/NUTS_RG_20M_2021_3035.shp/"
loc_pop = folder+"Data/Population/"
loc_out = folder+"Data/E-OBS/"

# read shapefile
shp = gp.read_file(loc_shp).loc[:,["NUTS_ID","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID","geometry"]]
print(shp)

# read tmax and tmin
f_x = np.sort([loc_tx+x for x in os.listdir(loc_tx) if "v30" in x])
f_n = np.sort([loc_tn+x for x in os.listdir(loc_tn) if "v30" in x])

tx_in = xr.open_mfdataset(f_x,concat_dim="time",combine="nested").tx #.load()
tn_in = xr.open_mfdataset(f_n,concat_dim="time",combine="nested").tn #.load()

latmin = 20
latmax = 80
lonmin = -20
lonmax = 60

# read population
population = xr.open_dataset(loc_pop+"population_weights_precip.nc").data_vars["layer"]
pop = population[::-1,:].loc[latmin:latmax,lonmin:lonmax]

# years
y1 = 1980
y2 = 2023

calc_temp = False
if calc_temp:
    # loop through years, do calculation
    for y in np.arange(y1,y2+1,1):
        print(y)
        tx_y = tx_in.loc[str(y)+"-01-01":str(y)+"-12-31",latmin:latmax,lonmin:lonmax].load()
        #tn_y = tn_in.loc[str(y)+"-01-01":str(y)+"-12-31",latmin:latmax,lonmin:lonmax].load()
        
        lat = tx_y.latitude
        lon = tx_y.longitude
        if y==y1:
            pop_regrid = pop.interp(latitude=lat,longitude=lon)
        
        # daily tx for each nuts3 region
        weightmap = xa.pixel_overlaps(tx_y,shp,weights=pop_regrid)
        tx_nuts3_y = xa.aggregate(tx_y,weightmap).to_dataframe().reset_index().drop(columns="poly_idx")
        
        ## add to big df
        if y==y1:
            df_out_tx = tx_nuts3_y.copy()
        else:
            df_out_tx = pd.concat([df_out_tx,tx_nuts3_y])
            
            
    df_out_tx.to_csv(loc_out+"e-obs_nuts3_daily_tx_popweight_"+str(y1)+"-"+str(y2)+".csv")
    print(loc_out+"e-obs_nuts3_daily_tx_popweight_"+str(y1)+"-"+str(y2)+".csv")
    
    
    # loop through years, do calculation for tn
    del(df_out_tx)
    del(tx_nuts3_y)
    #del(tx_in)
    
    for y in np.arange(y1,y2+1,1):
        print(y)
        tn_y = tn_in.loc[str(y)+"-01-01":str(y)+"-12-31",latmin:latmax,lonmin:lonmax].load()
        
        lat = tn_y.latitude
        lon = tn_y.longitude
        if y==y1:
            pop_regrid = pop.interp(latitude=lat,longitude=lon)
        
        # daily tn for each nuts3 region
        weightmap = xa.pixel_overlaps(tn_y,shp,weights=pop_regrid)
        tn_nuts3_y = xa.aggregate(tn_y,weightmap).to_dataframe().reset_index().drop(columns="poly_idx")
        
        ## add to big df
        if y==y1:
            df_out_tn = tn_nuts3_y.copy()
        else:
            df_out_tn = pd.concat([df_out_tn,tn_nuts3_y])
            
        
    
    df_out_tn.to_csv(loc_out+"e-obs_nuts3_daily_tn_popweight_"+str(y1)+"-"+str(y2)+".csv")
    print(loc_out+"e-obs_nuts3_daily_tn_popweight_"+str(y1)+"-"+str(y2)+".csv")


calc_precip = True
if calc_precip:
    pr_in = xr.open_dataset(loc_pr+"rr_ens_mean_0.1deg_reg_v30.0e.nc")
    for y in np.arange(y1,y2+1,1):
        print(y)
        pr_y = pr_in.rr.loc[str(y)+"-01-01":str(y)+"-12-31",latmin:latmax,lonmin:lonmax].load()
        
        lat = pr_y.latitude
        lon = pr_y.longitude
        if y==y1:
            pop_regrid = pop.interp(latitude=lat,longitude=lon)
        
        # daily tx for each nuts3 region
        pr_y.name = "pr"
        weightmap = xa.pixel_overlaps(pr_y,shp,weights=pop_regrid)
        pr_nuts3_y = xa.aggregate(pr_y,weightmap).to_dataframe().reset_index().drop(columns="poly_idx")
        
        ## add to big df
        if y==y1:
            df_out_pr = pr_nuts3_y.copy()
        else:
            df_out_pr = pd.concat([df_out_pr,pr_nuts3_y])

    df_out_pr.to_csv(loc_out+"e-obs_nuts3_daily_precip_popweight_"+str(y1)+"-"+str(y2)+".csv")
    print(loc_out+"e-obs_nuts3_daily_precip_popweight_"+str(y1)+"-"+str(y2)+".csv")



### reaggregate the same things but using ISO8601 standard weeks
### daily mean temperature specifically
### following Ballester et al etc
### as well as the summed polynomials following carleton et al 
### and the summed daily lagged interactions

## read in 
tx_in = pd.read_csv(loc_out+"e-obs_nuts3_daily_tx_popweight_"+str(y1)+"-"+str(y2)+".csv",index_col=0)
tn_in = pd.read_csv(loc_out+"e-obs_nuts3_daily_tn_popweight_"+str(y1)+"-"+str(y2)+".csv",index_col=0)

tn = tn_in.loc[:,["time","NUTS_ID","tn"]]
del(tn_in)

t = pd.merge(tx_in,tn,how="left",on=["time","NUTS_ID"])

pr_in = pd.read_csv(loc_out+"e-obs_nuts3_daily_precip_popweight_"+str(y1)+"-"+str(y2)+".csv",index_col=0)
t = pd.merge(t,pr_in.loc[:,["time","NUTS_ID","pr"]],how="left",on=["time","NUTS_ID"])
del(pr_in)
del(tx_in)

t["tm"] = (t.tx.values + t.tn.values)/2.0
t["tm2"] = t.tm**2
t["tm3"] = t.tm**3
t["tm4"] = t.tm**4
t["tx2"] = t.tx**2
t["tx3"] = t.tx**3
t["tx4"] = t.tx**4
t["tn2"] = t.tn**2
t["tn3"] = t.tn**3
t["tn4"] = t.tn**4
t["pr2"] = t.pr**2
t["time"] = pd.to_datetime(t["time"])
t["week"] = t.time.dt.isocalendar().week 
t["year"] = t.time.dt.year.values
t["doy"] = t.time.dt.dayofyear.values
t["year-week"] = [str(x)+"-"+str(y) for x, y in zip(t.year.values,t.week.values)]
#print(t)
#sys.exit()

## calculate climatology and anoms
t_clm = t.loc[:,["NUTS_ID","doy","tm"]].groupby(["NUTS_ID","doy"]).mean().reset_index().rename(columns={"tm":"tm_clm"})
t_clm_pre2003 = t.loc[t.year<2003,["NUTS_ID","doy","tm"]].groupby(["NUTS_ID","doy"]).mean().reset_index()
t = pd.merge(t,t_clm,how="left",on=["NUTS_ID","doy"])
t["tm_anom"] = t.tm.values - t.tm_clm.values
t["tm_anom"] = t["tm_anom"].replace(0, np.nan)
#t["tm_anom"]  = t.groupby('NUTS_ID')["tm_anom"].filter(lambda x: ~(x.isna().any()))
t_anom_sum = t[["NUTS_ID","tm"]].groupby("NUTS_ID").sum().reset_index()
t_nuts_keep = t_anom_sum.loc[t_anom_sum.tm.values>0,:]
t = t.loc[[x in t_nuts_keep.NUTS_ID.values for x in t.NUTS_ID.values],:]

## sort, lag anomalies
t = t.sort_values(by=["NUTS_ID","time"]).reset_index().drop(columns="index")
t["tm_anom_l1"] = t.groupby("NUTS_ID")["tm_anom"].shift(periods=1)
lbs = ["1","2","3","4","5"] #,"6","7","8","9","10"]
print(t)
print(t.groupby("NUTS_ID")["tm_anom_l1"].apply(lambda x: pd.qcut(x,q=len(lbs),labels=lbs)).reset_index())
t["tm_anom_l1_bin"] = t.groupby("NUTS_ID")["tm_anom_l1"].apply(lambda x: pd.qcut(x,q=len(lbs),labels=lbs)).values
print(t)
#sys.exit()
#t["tm_anom_l1_bin"] = pd.qcut(x=t.tm_anom_l1,q=4,labels=["1","2","3","4"])
t.to_csv(loc_out+"e-obs_nuts3_daily_tmean_popweight_"+str(y1)+"-"+str(y2)+".csv")

# small version
y1_small = 1994
y2_small = 2023
tsmall = t.loc[(t.year>=y1_small)&(t.year<=y2_small),:]
tsmall.to_csv(loc_out+"e-obs_nuts3_daily_tmean_popweight_"+str(y1_small)+"-"+str(y2_small)+".csv")

## weekly
tpoly_weekly = t.loc[:,["NUTS_ID","LEVL_CODE","CNTR_CODE","NUTS_NAME","year-week","year","week","tm","tm2","tm3","tm4",\
                        "tx","tx2","tx3","tx4","tn","tn2","tn3","tn4","pr","pr2"]].groupby(["NUTS_ID","LEVL_CODE","CNTR_CODE","NUTS_NAME","year-week","year","week"]).sum().reset_index()
tmean_weekly = t.loc[:,["NUTS_ID","year","week","tm"]].groupby(["NUTS_ID","year","week"]).mean().reset_index()
tpoly_weekly = tpoly_weekly.rename(columns={"tm":"tpoly1","tm2":"tpoly2","tm3":"tpoly3","tm4":"tpoly4","tx":"txpoly1","tx2":"txpoly2","tx3":"txpoly3","tx4":"txpoly4",\
                                            "tn":"tnpoly1","tn2":"tnpoly2","tn3":"tnpoly3","tn4":"tnpoly4","pr":"prpoly1","pr2":"prpoly2"})
tpoly_weekly_out = pd.merge(tpoly_weekly,tmean_weekly,on=["NUTS_ID","year","week"],how="left")
tpoly_weekly_out.to_csv(loc_out+"e-obs_nuts3_weekly_temperature_precip_popweight_"+str(y1)+"-"+str(y2)+".csv")
print(loc_out+"e-obs_nuts3_weekly_temperature_precip_popweight_"+str(y1)+"-"+str(y2)+".csv")




