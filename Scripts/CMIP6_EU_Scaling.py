# NUTS3-level long-term mean temperature in CMIP6

# packages
import xarray as xr
import numpy as np
import sys
import os
import datetime
import pandas as pd
import xagg as xa
import geopandas as gp
import xesmf as xe
from functools import reduce

# locations
folder = "/your/path/to/project/folder/"
loc_cmip6_hist = "/path/to/historical/cmip6/monthly/tas/"
loc_cmip6_scenarios = "/path/to/scenariomip/data/"
loc_pop = folder+"Data/Population/"
loc_shp = folder+"Data/NUTS_RG_20M_2021_3035.shp/"
loc_out = folder+"Data/CMIP6/"
loc_out_combined = folder+"Data/CMIP6/"

# lat and lon bounds for europe 
latmin = 20
latmax = 80
lonmin = -20
lonmax = 60

# read shapefile
shp = gp.read_file(loc_shp).loc[:,["NUTS_ID","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID","geometry"]]
print(shp)

# read population
population = xr.open_dataset(loc_pop+"population_grid.nc").data_vars["layer"]
pop = population[::-1,:].loc[latmin:latmax,:].rename({"latitude":"lat","longitude":"lon"})

# years
y1_hist = 1850
y2_hist = 2014
y1_ssp = 2015
y2_ssp = 2100
y1_clm = 1850
y2_clm = 1900

# monthly to yearly
def monthly_to_yearly_mean(x):
        # calculate annual mean from monthly data
        # after weighting for the difference in month length
        # x must be data-array with time coord
        # xarray must be installed
        # x_yr = x.resample(time="YS").mean(dim="time") is wrong
        # because it doesn't weight for the # of days in each month
        days_in_mon = x.time.dt.days_in_month
        wgts = days_in_mon.groupby("time.year")/days_in_mon.groupby("time.year").sum()
        ones = xr.where(x.isnull(),0.0,1.0)
        x_sum = (x*wgts).resample(time="YS").sum(dim="time")
        ones_out = (ones*wgts).resample(time="YS").sum(dim="time")
        return(x_sum/ones_out)
        

# new grid
res = 2
lon_new = np.arange(1,359+res,res)
lat_new = np.arange(-89,89+res,res)
latshape = len(lat_new)
lonshape = len(lon_new)
# spatial weights for gmt
wgt = xr.DataArray(np.zeros((latshape,lonshape)),
                coords=[lat_new,lon_new],dims=["lat","lon"])
for ll in np.arange(0,lonshape,1):
    wgt[:,ll] = np.cos(np.radians(lat_new))

# new grid #2
res2 = 0.1
grid_out_eu = xr.Dataset({"lat": (["lat"], np.arange(latmin,latmax+res2,res2), {"units": "degrees_north"}),
                       "lon": (["lon"], np.arange(lonmin,lonmax+res2,res2), {"units": "degrees_east"})})
pop_regrid = pop.interp(lat=grid_out_eu.lat.values,lon=grid_out_eu.lon.values)
#print(pop_regrid)

# loop through scenarios
scenarios = ["ssp370"] 
for e in scenarios:
    print(e)
    
    # hist models
    hist_models = np.array([x for x in sorted(os.listdir(loc_cmip6_hist)) if (x.endswith(".nc"))])
    hist_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in hist_models])

    # the specific scenario
    ssp_models = np.array([x for x in sorted(os.listdir(loc_cmip6_scenarios+e+"/mon/tas/")) if (x.endswith(".nc"))])
    ssp_models_prefix = np.array([x.split("_")[2]+"_"+x.split("_")[4] for x in ssp_models])
    
    # overlapping set of models
    models_all = np.array(reduce(np.intersect1d,(hist_models_prefix,ssp_models_prefix)))
    # limit to one per model name
    modelnames = np.array([x.split("_")[0] for x in models_all])
    modelnames_uq = np.unique(modelnames)
    models = []
    for x in modelnames_uq:
        models_x = models_all[modelnames==x]
        if len(models_x[["r1i1p1f1" in z for z in models_x]])>0:
            models.append(models_x[["r1i1p1f1" in z for z in models_x]][0])
        else:
            models.append(models_x[0])
            
    #models_final = []
    already_finished = np.array([x.split("_")[0] for x in os.listdir(loc_out) if (e in x)&(str(y2_ssp) in x)])
    for m in models:
        print(m,flush=True)
        mname = m.split("_")[0]
        mreal = m.split("_")[1]
        
        if (mname in already_finished)|("AWI" in m)|("EC-Earth3" in m)|("GFDL" in m)|("MIROC6" in m): #|("EC-Earth3" in m)|("FGOALS" in m):
            continue

        #if (m in already_finished): #|("ACCESS" in m)|("MIROC" in m):
        #    continue

        model_hist = mname+"_historical_"+mreal
        model_ssp = mname+"_"+e+"_"+mreal
        
        # read tas hist
        print("reading hist data",flush=True)
        #print(model_hist)
        tas_ds_hist = xr.open_mfdataset(loc_cmip6_hist+"tas_Amon"+"_"+model_hist+"*.nc",concat_dim="time",combine="nested")
        tm_hist = tas_ds_hist.coords["time"].load()
        if tm_hist.dt.year[0] > y1_hist:
            continue
        tas_hist = tas_ds_hist.tas[tm_hist.dt.year<=y2_hist,:,:].load().drop_duplicates(dim="time",keep="first")
        if tas_hist.max()>200:
            tas_hist = tas_hist-273.15
        
        #print(tas_hist)
        #sys.exit()
        print("reading ssp data",flush=True)
        tas_ds_ssp = xr.open_mfdataset(loc_cmip6_scenarios+e+"/mon/tas/"+"tas_Amon"+"_"+model_ssp+"*.nc",concat_dim="time",combine="nested")
        tm_ssp = tas_ds_ssp.coords["time"].load()
        if tm_ssp.dt.year[-1] < y2_ssp:
            continue
        tas_ssp = tas_ds_ssp.tas.load().drop_duplicates(dim="time",keep="first")
        if tas_ssp.max()>200:
            tas_ssp = tas_ssp-273.15
        
        tas = xr.concat([tas_hist,tas_ssp],dim="time")
        del([tas_hist,tas_ssp])
        
        # calculate ann
        tas_ann = monthly_to_yearly_mean(tas)
        del(tas)
        #sys.exit()
        
        print("calculating GMST")
        # synchronize coords and regrid
        if "latitude" in tas_ann.coords:
            tas_ann = tas_ann.rename({"latitude":"lat","longitude":"lon"})
        if np.amin(tas_ann.coords["lon"].values)<0:
            tas_ann.coords["lon"] = tas_ann.coords["lon"].values % 360
        tas_ann_regrid = tas_ann.interp(lat=lat_new,lon=lon_new)
        
        # gmst
        gmst = tas_ann_regrid.weighted(wgt).mean(dim=["lat","lon"])
        gmst.coords["time"] = np.arange(y1_hist,y2_ssp+1,1)
        gmst_anom = gmst - gmst.loc[y1_clm:y2_clm].mean(dim="time")
        
        ## now aggregating to EU nuts regions
        print("regridding to finer res for EU regions and aggregating")
        # flip longitude
        tas_ann.coords['lon'] = (tas_ann.coords['lon'] + 180) % 360 - 180
        tas_ann = tas_ann.sortby(tas_ann.lon)
        
        # regrid to higher res but just make the higher res cells the same values
        # (using nearest_s2d)
        # so that the aggregation to counties works
        regridder = xe.Regridder(tas_ann,grid_out_eu,method="nearest_s2d")
        tas_ann_regrid2 = regridder(tas_ann)
        
        # now aggregate to nuts3 regions
        tas_ann_regrid2.name = "tmean"
        weightmap = xa.pixel_overlaps(tas_ann_regrid2,shp,weights=pop_regrid)
        tas_nuts3 = xa.aggregate(tas_ann_regrid2,weightmap).to_dataset(loc_dim="nuts")
        tas_nuts3.coords["nuts"] = tas_nuts3.NUTS_ID
        tas_nuts3_var = tas_nuts3.tmean
        tas_nuts3_var.coords["time"] = np.arange(y1_hist,y2_ssp+1,1)
        
        ## write dataset
        tmean_ds = xr.Dataset({"gmst":(["time"],gmst.values),
                               "temp_nuts":(["nuts","time"],tas_nuts3_var.values)},
                                coords={"time":(["time"],gmst.time.values),
                                        "nuts":(["nuts"],tas_nuts3_var.nuts.values)})
                                         
        tmean_ds.attrs["creation_date"] = str(datetime.datetime.now())
        tmean_ds.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
        tmean_ds.attrs["variable_description"] = "global mean temperature and nuts3-level mean temperature from CMIP6 models"
        tmean_ds.attrs["created_from"] = os.getcwd()+"/CMIP6_EU_Scaling.py"
            
        fname_out = loc_out+mname+"_historical_"+e+"_europe_nuts3_mean_temperature_popweight_"+str(y1_hist)+"-"+str(y2_ssp)+".nc"
        tmean_ds.to_netcdf(fname_out,mode="w")
        print(fname_out,flush=True)
    
    ## all models
    all_projections = xr.open_mfdataset([loc_out+x for x in sorted(os.listdir(loc_out)) if (e in x)],combine="nested",concat_dim="model")
    models_final = [x.split("_")[0] for x in sorted(os.listdir(loc_out)) if (e in x)]
    all_projections.coords["model"] = models_final
    
    all_projections.attrs["creation_date"] = str(datetime.datetime.now())
    all_projections.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
    all_projections.attrs["variable_description"] = "global mean temperature and nuts3-level mean temperature from CMIP6 models"
    all_projections.attrs["created_from"] = os.getcwd()+"/CMIP6_EU_Scaling.py"
            
    fname_out = loc_out_combined+"CMIP6_historical_"+e+"_europe_nuts3_mean_temperature_popweight_"+str(y1_hist)+"-"+str(y2_ssp)+".nc"
    all_projections.to_netcdf(fname_out,mode="w")
    print(all_projections)
    print(fname_out,flush=True)
    
        
