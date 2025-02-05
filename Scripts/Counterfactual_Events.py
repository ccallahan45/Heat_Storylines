# counterfactual heat waves using CNN predictions
#### Christopher Callahan
#### christophercallahan@stanford.edu


import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask as rm
import pandas as pd
import xagg as xa
import sys
import os
import cartopy as cart
from rasterio import features
from affine import Affine
import descartes
from scipy import stats
import geopandas as gp
from tqdm import tqdm
import datetime

# data locations
folder = "/your/path/to/project/folder/"
loc_cnn = folder+"Data/Trok_Predictions/"
loc_events = folder+"Data/Events/"
loc_era5 = folder+"Data/Events/"
loc_out = folder+"Data/Events/"
loc_shp = folder+"Data/NUTS_RG_20M_2021_3035.shp/"
loc_pop = folder+"Data/Population/"


loc_tx = "/path/to/eobs/dailymax/temp/"
loc_tn = "/path/to/eobs/dailymin/temp/"



# set up events
events_df = pd.read_csv(loc_events+"event_definitions.csv")
hw = events_df.event.values
begin = pd.to_datetime(events_df.begindate).dt.strftime('%Y-%m-%d')
end = pd.to_datetime(events_df.enddate).dt.strftime('%Y-%m-%d')
print(events_df)


cnns = [1,2,3]
t_cnn_wce = xr.open_mfdataset(loc_cnn+"CEU_CNN_PREDICTIONS_*",combine="nested",concat_dim="cnn")
t_cnn_wce.coords["cnn"] = cnns
t_cnn_neu = xr.open_mfdataset(loc_cnn+"NEU_CNN_PREDICTIONS_*",combine="nested",concat_dim="cnn")
t_cnn_neu.coords["cnn"] = cnns
t_cnn_med = xr.open_mfdataset(loc_cnn+"MED_CNN_PREDICTIONS_*",combine="nested",concat_dim="cnn")
t_cnn_med.coords["cnn"] = cnns

t_diff_wce = t_cnn_wce.counterfactual_tmean.load() - t_cnn_wce.predicted_tmean.load()
t_diff_neu = t_cnn_neu.counterfactual_tmean.load() - t_cnn_neu.predicted_tmean.load()
t_diff_med = t_cnn_med.counterfactual_tmean.load() - t_cnn_med.predicted_tmean.load()
gmts = t_diff_wce.gmt.values

# regions
ar6_regions_gp = rm.defined_regions.ar6.land.to_geodataframe().reset_index()
ar6_eu = ar6_regions_gp.loc[[x in ["NEU","WCE","MED"] for x in ar6_regions_gp.abbrevs.values],:]

# go through ERA5 maps of events and add additional temperature on top of them
# mask each map to each region
# then match NN projection to the grid points in each mask
era5_events = xr.open_dataset(loc_era5+"ERA5_event_temperature_hgt_soilmoisture_anomalies.nc")
elat = era5_events.lat.values
elon = era5_events.lon.values
era5_mask = rm.mask_geopandas(ar6_eu,elon,elat)

# build empty data
cf_events = xr.DataArray(np.full((len(hw),len(gmts),len(elat),len(elon)),np.nan),
                        coords=[hw,gmts,elat,elon],dims=["event","gmt","lat","lon"])
cf_events.coords["begindate"] = xr.DataArray(begin,coords=[hw],dims=["event"])
cf_events.coords["enddate"] = xr.DataArray(end,coords=[hw],dims=["event"])

## loop through events, get the deltas for the right dates
## project delta onto a map
## add to obs map

for h in np.arange(0,len(hw),1):
    event = hw[h]
    print(event)
    begindate = begin[h]
    enddate = end[h]
    
    era5_event = era5_events.temp_anomalies.loc[event,:,:].load()
    t_diff_wce_event = t_diff_wce.loc[:,:,begindate:enddate].mean(dim=["cnn","time"])
    t_diff_neu_event = t_diff_neu.loc[:,:,begindate:enddate].mean(dim=["cnn","time"])
    t_diff_med_event = t_diff_med.loc[:,:,begindate:enddate].mean(dim=["cnn","time"])
    
    # broadcast cnn differences to lat and lon grid
    # then mask just to each region
    t_diff_wce_event_grid = t_diff_wce_event.expand_dims(lat=elat,lon=elon)
    t_diff_neu_event_grid = t_diff_neu_event.expand_dims(lat=elat,lon=elon)
    t_diff_med_event_grid = t_diff_med_event.expand_dims(lat=elat,lon=elon)
    
    ## NEU = 16, WCE/CEU = 17, MED = 19
    t_diff_wce_event_grid_mask = t_diff_wce_event_grid.where(era5_mask==17,other=0.0)
    t_diff_neu_event_grid_mask = t_diff_neu_event_grid.where(era5_mask==16,other=0.0)
    t_diff_med_event_grid_mask = t_diff_med_event_grid.where(era5_mask==19,other=0.0)
    
    # counterfactual event
    cf_event = era5_event + t_diff_wce_event_grid_mask + t_diff_neu_event_grid_mask + t_diff_med_event_grid_mask
    
    # add to array
    cf_events.loc[event,:,:,:] = cf_event.transpose("gmt","lat","lon").values

# write out
events_out = xr.Dataset({"counterfactual_events":(["event","gmt","lat","lon"],cf_events.values)},
                            coords={"event":(["event"],hw),
                                    "gmt":(["gmt"],gmts),
                                    "lat":(["lat"],elat),
                                    "lon":(["lon"],elon),
                                    "begindate":(["begindate"],begin),
                                    "enddate":(["enddate"],end)})
events_out.attrs["creation_date"] = str(datetime.datetime.now())
events_out.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
events_out.attrs["variable_description"] = "counterfactual heat wave maps created by adding CNN predictions in IPCC regions to ERA5 maps"
events_out.attrs["created_from"] = os.getcwd()+"/Counterfactual_Events.py"

fname_out = loc_out+"ERA5plusCNN_counterfactual_heatwave_maps.nc"
events_out.to_netcdf(fname_out,mode="w")
print(fname_out,flush=True)


## now do a similar thing for the E-OBS data
## add counterfactuals to each grid and day
## then re-aggregate

# shapefile for aggregation
shp = gp.read_file(loc_shp).loc[:,["NUTS_ID","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID","geometry"]]
print(shp)
latmin = 20
latmax = 80
lonmin = -20
lonmax = 60

# eobs 
f_x = np.sort([loc_tx+x for x in os.listdir(loc_tx) if "v30" in x])
f_n = np.sort([loc_tn+x for x in os.listdir(loc_tn) if "v30" in x])
tx_in = xr.open_mfdataset(f_x,concat_dim="time",combine="nested").tx #.load()
tn_in = xr.open_mfdataset(f_n,concat_dim="time",combine="nested").tn #.load()

# read population
population = xr.open_dataset(loc_pop+"population_grid.nc").data_vars["layer"]
pop = population[::-1,:].loc[latmin:latmax,lonmin:lonmax]

# gmts of interest
gmts = [0,1.5,2,3,4]

calc_overall_counterfactual = False
y1_eobs_clm = 1980
y2_eobs_clm = 2023
if calc_overall_counterfactual:
    # read eobs
    print("reading full e-obs data",flush=True)
    eobs_time = tx_in.time.load()
    summer_ind = (eobs_time.dt.month>=6)&(eobs_time.dt.month<=9)&(eobs_time.dt.year>=y1_eobs_clm)&(eobs_time.dt.year<=y2_eobs_clm)
    tx_all = tx_in[summer_ind,:,:].loc[:,latmin:latmax,lonmin:lonmax].load().rename({"latitude":"lat","longitude":"lon"})
    tn_all = tn_in[summer_ind,:,:].loc[:,latmin:latmax,lonmin:lonmax].load().rename({"latitude":"lat","longitude":"lon"})
    tm_all = (tx_all+tn_all)/2.0
    del([tx_all,tn_all])
    
    # get grid
    eolat = tm_all.lat.values
    eolon = tm_all.lon.values
    eobs_mask = rm.mask_geopandas(ar6_eu,eolon,eolat)
    
    print("masking CNN deltas",flush=True)
    # calculate the counterfactual for each day -- long-term average at 0 C
    # NEU = 16, WCE/CEU = 17, MED = 19
    t_diff_wce_grid_mask = t_diff_wce.mean(dim="cnn").sel(gmt=0).expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon").where(eobs_mask==17,other=0.0)
    t_diff_neu_grid_mask = t_diff_neu.mean(dim="cnn").sel(gmt=0).expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon").where(eobs_mask==16,other=0.0)
    t_diff_med_grid_mask = t_diff_med.mean(dim="cnn").sel(gmt=0).expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon").where(eobs_mask==19,other=0.0)
    
    #print(tm_all)
    #print(t_diff_wce_grid_mask)
    print("calculating every day's counterfactual from observations",flush=True)
    time_overlap = [x for x in tm_all.time.values if x in t_diff_wce_grid_mask.time.values]
    tm_all_cf = tm_all.loc[time_overlap,:,:] + t_diff_wce_grid_mask.loc[time_overlap,:,:] + t_diff_neu_grid_mask.loc[time_overlap,:,:] + t_diff_med_grid_mask.loc[time_overlap,:,:]
    #print(tm_all_cf)
    tm_all_cf_byday = tm_all_cf.groupby("time.dayofyear").mean(dim="time")
    #print(tm_all_cf_byday)
    
    del(tm_all_cf)
    
    print("aggregating to subnational regions",flush=True)
    # aggregate counterfactual to regions
    tm_all_cf_byday.name = "tmclm"
    pop_regrid = pop.interp(latitude=eolat,longitude=eolon)
    weightmap = xa.pixel_overlaps(tm_all_cf_byday,shp,weights=pop_regrid)
    cf_all_df = xa.aggregate(tm_all_cf_byday,weightmap).to_dataframe().reset_index().drop(columns=["poly_idx","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID"])
    #print(cf_all_df)
    #print(cf_all_df.loc[(["FR" in x for x in cf_all_df.NUTS_ID.values])&(cf_all_df.dayofyear==155),:])
    #print(cf_all_df.loc[(["ES" in x for x in cf_all_df.NUTS_ID.values])&(cf_all_df.dayofyear==200),:])
    #print(cf_all_df.loc[(["UK" in x for x in cf_all_df.NUTS_ID.values])&(cf_all_df.dayofyear==210),:])
    #sys.exit()
    del([tm_all,tm_all_cf_byday,t_diff_wce_grid_mask,t_diff_neu_grid_mask,t_diff_med_grid_mask])
    # deleting this to save memory
    # we'll just read little chunks in for each event
    print(cf_all_df)
    cf_all_df_out = cf_all_df.rename(columns={"NUTS_ID":"nuts"})
    fname = loc_out+"eobs_dayofyear_summer_climatology_"+str(y1_eobs_clm)+"-"+str(y2_eobs_clm)+".csv"
    cf_all_df_out.to_csv(fname)
    print(fname)
    #sys.exit()

print("now for individual events",flush=True)
# loop through each event
for h in np.arange(0,len(hw),1):
    event = hw[h]
    print(event,flush=True)
    begindate = begin[h]
    enddate = end[h]
    eventyr = int(begindate.split("-")[0])
    #print(eventyr)
    
    # add some padding on both sides to make sure our aggregation to the weekly level
    # is actually seven days and not less than seven
    begindate_padded = (pd.to_datetime(begindate) - pd.to_timedelta(30,unit='d')).strftime('%Y-%m-%d')
    enddate_padded = (pd.to_datetime(enddate) + pd.to_timedelta(30,unit='d')).strftime('%Y-%m-%d')
    
    # eobs for that time period
    #tm_event = tm_all.loc[begindate_padded:enddate_padded,:,:]
    #print(tm_event)
    tx_event = tx_in.loc[begindate_padded:enddate_padded,latmin:latmax,lonmin:lonmax].load().rename({"latitude":"lat","longitude":"lon"})
    tn_event = tn_in.loc[begindate_padded:enddate_padded,latmin:latmax,lonmin:lonmax].load().rename({"latitude":"lat","longitude":"lon"})
    tm_event = (tx_event + tn_event)/2.0
    #print(tm_event)
    del([tx_event,tn_event])
    
    # get grid
    eolat = tm_event.lat.values
    eolon = tm_event.lon.values
    eobs_mask = rm.mask_geopandas(ar6_eu,eolon,eolat)
    
    # predictions for each region for that time period
    t_diff_wce_event = t_diff_wce.loc[:,gmts,begindate:enddate] 
    t_diff_neu_event = t_diff_neu.loc[:,gmts,begindate:enddate]
    t_diff_med_event = t_diff_med.loc[:,gmts,begindate:enddate]
    #t_diff_wce_event = t_diff_wce.loc[gmts,begindate:enddate] 
    #t_diff_neu_event = t_diff_neu.loc[gmts,begindate:enddate]
    #t_diff_med_event = t_diff_med.loc[gmts,begindate:enddate]
    # broadcast cnn differences to lat and lon grid
    # then mask just to each region
    t_diff_wce_event_grid = t_diff_wce_event.expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon","gmt","cnn")
    t_diff_neu_event_grid = t_diff_neu_event.expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon","gmt","cnn")
    t_diff_med_event_grid = t_diff_med_event.expand_dims(lat=eolat,lon=eolon).transpose("time","lat","lon","gmt","cnn")
    # NEU = 16, WCE/CEU = 17, MED = 19
    t_diff_wce_event_grid_mask = t_diff_wce_event_grid.where(eobs_mask==17,other=0.0)
    t_diff_neu_event_grid_mask = t_diff_neu_event_grid.where(eobs_mask==16,other=0.0)
    t_diff_med_event_grid_mask = t_diff_med_event_grid.where(eobs_mask==19,other=0.0)
    
    # create new vars wih the expanded time coordinates, add our predictions for just the specific
    # event dates into them
    # so we can sum them with the time-padded obs data
    
    t_diff_wce_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts),len(cnns))),
                                    coords=[tm_event.time,eolat,eolon,gmts,cnns],dims=["time","lat","lon","gmt","cnn"])
    t_diff_neu_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts),len(cnns))),
                                    coords=[tm_event.time,eolat,eolon,gmts,cnns],dims=["time","lat","lon","gmt","cnn"])
    t_diff_med_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts),len(cnns))),
                                    coords=[tm_event.time,eolat,eolon,gmts,cnns],dims=["time","lat","lon","gmt","cnn"])
    t_diff_wce_extended.loc[begindate:enddate,:,:,:,:] = t_diff_wce_event_grid_mask.loc[begindate:enddate,:,:,:,:]
    t_diff_neu_extended.loc[begindate:enddate,:,:,:,:] = t_diff_neu_event_grid_mask.loc[begindate:enddate,:,:,:,:]
    t_diff_med_extended.loc[begindate:enddate,:,:,:,:] = t_diff_med_event_grid_mask.loc[begindate:enddate,:,:,:,:]
    """
    t_diff_wce_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts))),
                                    coords=[tm_event.time,eolat,eolon,gmts],dims=["time","lat","lon","gmt"])
    t_diff_neu_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts))),
                                    coords=[tm_event.time,eolat,eolon,gmts],dims=["time","lat","lon","gmt"])
    t_diff_med_extended = xr.DataArray(np.zeros((len(tm_event.time),len(eolat),len(eolon),len(gmts))),
                                    coords=[tm_event.time,eolat,eolon,gmts],dims=["time","lat","lon","gmt"])
    t_diff_wce_extended.loc[begindate:enddate,:,:,:] = t_diff_wce_event_grid_mask.loc[begindate:enddate,:,:,:]
    t_diff_neu_extended.loc[begindate:enddate,:,:,:] = t_diff_neu_event_grid_mask.loc[begindate:enddate,:,:,:]
    t_diff_med_extended.loc[begindate:enddate,:,:,:] = t_diff_med_event_grid_mask.loc[begindate:enddate,:,:,:]
    """
    
    # add delta to obs
    cf_event = tm_event + t_diff_wce_extended + t_diff_neu_extended + t_diff_med_extended
    cf_event.name = "tmcf"
    
    # aggregate to nuts regions
    pop_regrid = pop.interp(latitude=eolat,longitude=eolon)
    weightmap = xa.pixel_overlaps(cf_event,shp,weights=pop_regrid)
    cf_event_df = xa.aggregate(cf_event,weightmap).to_dataframe().reset_index().drop(columns=["poly_idx","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID"])
    
    # read in long-term climatology from previous block of code
    eobs_clm = pd.read_csv(loc_out+"eobs_dayofyear_summer_climatology_1980-2023.csv",index_col=0).rename(columns={"nuts":"NUTS_ID"})
    cf_event_df["dayofyear"] = cf_event_df.time.dt.dayofyear
    cf_event_df = pd.merge(cf_event_df,eobs_clm,how="left",on=["NUTS_ID","dayofyear"])
    # set clm outside of event boundaries to be equal to counterfactual
    cf_event_df["tmclm_alldays"] = cf_event_df.tmclm.values*1.0
    cf_event_df.loc[(cf_event_df.time<begindate)|(cf_event_df.time>enddate),"tmclm"] = cf_event_df.loc[(cf_event_df.time<begindate)|(cf_event_df.time>enddate),"tmcf"].values
    #cf_event_df['tmclm'] = cf_event_df['tmclm'].fillna(cf_event_df.tmcf)
    
    # aggregate obs to regions
    tm_event.name = "tmobs"
    weightmap = xa.pixel_overlaps(tm_event,shp,weights=pop_regrid)
    obs_event_df = xa.aggregate(tm_event,weightmap).to_dataframe().reset_index().drop(columns=["poly_idx","LEVL_CODE","CNTR_CODE","NUTS_NAME","FID"])
    
    ## write out both daily and weekly
    cf_event_daily = pd.merge(cf_event_df,obs_event_df,on=["time","NUTS_ID"])
    cf_event_daily.to_csv(loc_out+event+"_EOBSplusCNN_counterfactual_nuts3_daily_tmean.csv")
    print(loc_out+event+"_EOBSplusCNN_counterfactual_nuts3_daily_tmean.csv")
    
    # add polynomials 
    cf_event_df["tmcf2"] = cf_event_df.tmcf**2
    cf_event_df["tmcf3"] = cf_event_df.tmcf**3
    cf_event_df["tmcf4"] = cf_event_df.tmcf**4
    cf_event_df["tmclm2"] = cf_event_df.tmclm**2
    cf_event_df["tmclm3"] = cf_event_df.tmclm**3
    cf_event_df["tmclm4"] = cf_event_df.tmclm**4
    
    # time vars
    cf_event_df["year"] = pd.to_datetime(cf_event_df.time).dt.year
    cf_event_df["week"] = pd.to_datetime(cf_event_df.time).dt.isocalendar().week
    #cf_event_df["quantile"] = 0.5
    
    # agg to weekly and limit to three weeks on either side of the event
    # to accommodate lags in the mortality regression
    cf_event_wk = cf_event_df.drop(columns="time").groupby(["year","week","NUTS_ID","gmt","cnn"]).sum().reset_index() 
    cf_event_wk = cf_event_wk.rename(columns={"tmcf":"tcfpoly1","tmcf2":"tcfpoly2","tmcf3":"tcfpoly3","tmcf4":"tcfpoly4",
        "tmclm":"tclmpoly1","tmclm2":"tclmpoly2","tmclm3":"tclmpoly3","tmclm4":"tclmpoly4"})
    
    # get week of the dates of interest so we can limit the final output data to those weeks +/- 3
    # to accommodate lags from the regression
    beginweek = pd.to_datetime(begindate).isocalendar().week
    endweek = pd.to_datetime(enddate).isocalendar().week
    cf_event_wk_out = cf_event_wk.loc[(cf_event_wk.week>=(beginweek-3))&(cf_event_wk.week<=(endweek+4)),:]
    
    cf_event_wk_out.to_csv(loc_out+event+"_EOBSplusCNN_counterfactual_nuts3_weekly_tmean.csv")
    print(loc_out+event+"_EOBSplusCNN_counterfactual_nuts3_weekly_tmean.csv")
    
    
    ## now do a version of this averaged over each IPCC region for visualization
    tm_event_wce = tm_event.loc[begindate:enddate,:,:].where(eobs_mask==17,other=np.nan).mean(dim=["time","lat","lon"])
    tm_event_neu = tm_event.loc[begindate:enddate,:,:].where(eobs_mask==16,other=np.nan).mean(dim=["time","lat","lon"])
    tm_event_med = tm_event.loc[begindate:enddate,:,:].where(eobs_mask==19,other=np.nan).mean(dim=["time","lat","lon"])
    wce_event = tm_event_wce + t_diff_wce_event.mean(dim=["cnn","time"])
    neu_event = tm_event_neu + t_diff_neu_event.mean(dim=["cnn","time"])
    med_event = tm_event_med + t_diff_med_event.mean(dim=["cnn","time"])
    
    event_regions = wce_event.expand_dims("region")
    event_regions = xr.concat([event_regions,neu_event],dim="region")
    event_regions = xr.concat([event_regions,med_event],dim="region")
    event_regions.coords["region"] = ["WCE","NEU","MED"]
    if h==0:
        event_regions_all = event_regions.expand_dims("event")
    else:
        event_regions_all = xr.concat([event_regions_all,event_regions],dim="event")
event_regions_all.coords["event"] = hw
event_regions_all.name = "tm"
event_regions_all_df = event_regions_all.to_dataframe().reset_index()
event_regions_all_df.to_csv(loc_out+"events_EOBSplusCNN_IPCCregion_mean_temperature.csv")
print(event_regions_all_df)
print(loc_out+"events_EOBSplusCNN_IPCCregion_mean_temperature.csv")

