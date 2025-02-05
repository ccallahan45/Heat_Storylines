# counterfactual event mortality

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import geopandas as gp
import pyreadr
import datetime
import os
from tqdm import tqdm
import sys

#folder = "/your/path/to/project/folder/"
folder = "/Users/christophercallahan/Dropbox (Personal)/Research/Projects/Heat_Storylines/Replication_NCC/"
loc_panel = folder+"Data/Panel/"
loc_shp = folder+"Data/NUTS_RG_60M_2021_3035.shp/"
loc_reg = folder+"Data/RegressionResults/"
loc_events = folder+"Data/Events/"
loc_cmip = folder+"Data/CMIP6/"
loc_gmt = folder+"Data/GMT/"

# do we want to incorporate adaptation or not?
## you'll need to run the script a couple times, switching these flags
## to run the various adaptation scenarios
adaptation = True
gmt_adaptation = False
if (adaptation)&(gmt_adaptation):
    adapt_str = "_gmtadaptation" # simplest approach to adaptation
elif (adaptation)&(gmt_adaptation==False):
    adapt_str = "_scaledadaptation" # pattern scaling approach to adaptation
else:
    adapt_str = "_noadaptation" # base case


no_event_counterfactual = True
if no_event_counterfactual:
    cfstr = "_noeventcf"
else:
    cfstr = ""



# Read coefficients
age = "total"
nlag = 3
lags = np.arange(0,nlag+1,1)
y1 = 2015
y2 = 2019
nboot = 500
boot = np.arange(1,nboot+1,1)
npoly = 4
poly = np.arange(1,npoly+1,1)
coefs_in = pd.read_csv(loc_reg+"europe_weekly_mortality_coefficients_tmean-interact_"+str(nlag)+"lag_"+age+"age_"+str(y1)+"-"+str(y2)+".csv")
coefs_df = coefs_in.loc[coefs_in.boot<=nboot,["term","estimate","boot"]]


# Panel
panel = pyreadr.read_r(loc_panel+"eurostat_eobs_weekly_mortality_temperature_2000-2022.rds")[None]
panel_nuts = np.unique(panel.nuts.values)
nuts_countries = panel[["nuts","country"]].dropna().groupby("nuts").agg("first").reset_index()

# Climatological stuff
y1_clm = 2015
y2_clm = 2019
panel_pop_clm = panel.loc[(panel.year>=y1_clm)&(panel.year<=y2_clm)&(panel.age==age),["nuts","week","year","population","n_deaths","rate"]]
panel_pclm = panel_pop_clm.groupby(["nuts","week"]).mean().reset_index().drop(columns="year") 

# temperature mean
y1_tclm = 2000
y2_tclm = 2019
panel_temp_clm = panel.loc[(panel.year>=y1_tclm)&(panel.year<=y2_tclm)&(panel.age==age),["nuts","tm"]] #,"tpoly1","tpoly2","tpoly3","tpoly4"]]
panel_tclm = panel_temp_clm.groupby(["nuts"]).mean().reset_index() #.drop(columns="year") 
#panel_tclm = panel_tclm.rename(columns={"tpoly1":"tclmpoly1","tpoly2":"tclmpoly2","tpoly3":"tclmpoly3","tpoly4":"tclmpoly4"})

# scaling coefficients for adaptation
adapt_coefs = xr.open_dataarray(loc_cmip+"CMIP6_nuts3_global_regional_linear_scaling.nc")

# calc 2000-2019 mean GMT anomaly relative to 1850-1900
hadcrut = pd.read_csv(loc_gmt+"HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv")
hadcrut["gmt"] = hadcrut["Anomaly (deg C)"] - hadcrut.loc[(hadcrut.Time>=1850)&(hadcrut.Time<=1900),"Anomaly (deg C)"].mean()
gmt_2000s = hadcrut.loc[(hadcrut.Time>=2000)&(hadcrut.Time<=2019),"gmt"].mean()

# stuff for monte carlo
n_mc = 500 #1000
mc = np.arange(1,n_mc+1,1)

# Loop through events, calculate deaths, write out
events = ["july_1994","august_2003","july_2006","june_2019","august_2023"]
for e in events:
    print(e)
    cf_events_all = pd.read_csv(loc_events+e+"_EOBSplusCNN_counterfactual_nuts3_weekly_tmean.csv",index_col=0)
    cf_events = cf_events_all.loc[[x in nuts_countries.nuts.values for x in cf_events_all.NUTS_ID.values],:]
    del(cf_events_all)
    cf_events = pd.merge(cf_events.rename(columns={"NUTS_ID":"nuts"}),nuts_countries,how="left",on=["nuts"])
    cf_events = pd.merge(cf_events,panel_pclm,how="left",on=["nuts","week"])
    cf_events = pd.merge(cf_events,panel_tclm,how="left",on=["nuts"])

    # indices
    nuts = np.unique(cf_events.nuts.values)
    week = np.unique(cf_events.week.values)
    cnns = np.unique(cf_events["cnn"].values)
    gmts = np.unique(cf_events.gmt.values)

    mortality = xr.DataArray(np.full((len(nuts),len(week),len(gmts),n_mc),np.nan),
                             coords=[nuts,week,gmts,mc],
                             dims=["nuts","week","gmt","sample"])
    mortality.name = "deaths"
    population = xr.DataArray(np.full((len(nuts)),np.nan),
                             coords=[nuts],dims=["nuts"])
    population.name = "population"
    
    for gg in np.arange(0,len(gmts),1):
        np.random.seed(120)
        g = gmts[gg]
        print(g)
        event_g = cf_events.loc[cf_events.gmt==g,:].copy()
        event_base = cf_events.loc[cf_events.gmt==0,:].copy()
        for p in poly:
            for l in np.arange(1,nlag+1,1):
                event_g.loc[:,"tcfpoly"+str(p)+"_lag"+str(l)] = event_g.groupby(["nuts","year","cnn"]).shift(l).loc[:,"tcfpoly"+str(p)]
                event_g.loc[:,"tclmpoly"+str(p)+"_lag"+str(l)] = event_g.groupby(["nuts","year","cnn"]).shift(l).loc[:,"tclmpoly"+str(p)]
                event_base.loc[:,"tcfpoly"+str(p)+"_lag"+str(l)] = event_base.groupby(["nuts","year","cnn"]).shift(l).loc[:,"tcfpoly"+str(p)]
                #event_base.loc[:,"tclmpoly"+str(p)+"_lag"+str(l)] = event_base.groupby(["nuts","year","quantile"]).shift(l).loc[:,"tclmpoly"+str(p)]
                #event_g.loc[:,"tobspoly"+str(p)+"_lag"+str(l)] = event_g.groupby(["nuts","year","quantile"]).shift(l).loc[:,"tobspoly"+str(p)]

        for n in tqdm(np.arange(1,n_mc+1,1)):
            #print(n)
            i = np.random.choice(boot,size=1)
            coefs_b = coefs_df.loc[coefs_df.boot.values==i,:]
            cnn = np.random.choice(cnns,size=1)
            event_n = event_g.loc[event_g["cnn"].values==cnn,:].copy()
            event_nbase = event_base.loc[event_base["cnn"].values==cnn,:].copy()
            #i = boot[ii]
            #coefs_b = coefs_df.loc[coefs_df.boot==i,:]
            if (adaptation)&(gmt_adaptation==False):
                adapt_coefs_n = adapt_coefs.loc[:,np.random.choice(adapt_coefs.sample.values,size=1),:]
                tm_change = adapt_coefs_n.loc["slope",:]*g + adapt_coefs_n.loc["intercept",:]
                tm_change_df = tm_change.to_dataframe().reset_index().drop(columns="sample").rename(columns={"coefs":"tm_change"})
                event_n = pd.merge(event_n,tm_change_df,how="left",on="nuts")
                event_nbase = pd.merge(event_nbase,tm_change_df,how="left",on="nuts")
            elif (adaptation)&(gmt_adaptation):
                event_n["tm_change"] = g - gmt_2000s
                event_nbase["tm_change"] = g - gmt_2000s 
            else:
                event_n["tm_change"] = 0.0
                event_nbase["tm_change"] = 0.0

            event_n["tmnew"] = event_n.tm + event_n.tm_change
            event_nbase["tmnew"] = event_nbase.tm + event_nbase.tm_change

            mort_effect_pct = np.zeros(event_n.shape[0])
            for l in lags:
                ll = str(l)
                if l == 0:
                    c1 = coefs_b.loc[coefs_b.term=="tpoly1","estimate"].values[0]
                    c2 = coefs_b.loc[coefs_b.term=="tpoly2","estimate"].values[0]
                    c3 = coefs_b.loc[coefs_b.term=="tpoly3","estimate"].values[0]
                    c4 = coefs_b.loc[coefs_b.term=="tpoly4","estimate"].values[0]
                    i1 = coefs_b.loc[coefs_b.term=="tpoly1:tmean","estimate"].values[0]
                    i2 = coefs_b.loc[coefs_b.term=="tpoly2:tmean","estimate"].values[0]
                    i3 = coefs_b.loc[coefs_b.term=="tpoly3:tmean","estimate"].values[0]
                    i4 = coefs_b.loc[coefs_b.term=="tpoly4:tmean","estimate"].values[0]

                    lograte_pred_cf = event_n["tcfpoly1"]*(c1 + i1*event_n["tmnew"]) + \
                                      event_n["tcfpoly2"]*(c2 + i2*event_n["tmnew"]) + \
                                      event_n["tcfpoly3"]*(c3 + i3*event_n["tmnew"]) + \
                                      event_n["tcfpoly4"]*(c4 + i4*event_n["tmnew"])
                    if no_event_counterfactual==False:
                        lograte_pred_clm = event_nbase["tcfpoly1"]*(c1 + i1*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly2"]*(c2 + i2*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly3"]*(c3 + i3*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly4"]*(c4 + i4*event_nbase["tmnew"])
                    else:
                        lograte_pred_clm = event_n["tclmpoly1"]*(c1 + i1*event_n["tmnew"]) + \
                                       event_n["tclmpoly2"]*(c2 + i2*event_n["tmnew"]) + \
                                       event_n["tclmpoly3"]*(c3 + i3*event_n["tmnew"]) + \
                                       event_n["tclmpoly4"]*(c4 + i4*event_n["tmnew"])    
                            
                else:
                    c1 = coefs_b.loc[coefs_b.term=="tpoly1_lag"+str(l),"estimate"].values[0]
                    c2 = coefs_b.loc[coefs_b.term=="tpoly2_lag"+str(l),"estimate"].values[0]
                    c3 = coefs_b.loc[coefs_b.term=="tpoly3_lag"+str(l),"estimate"].values[0]
                    c4 = coefs_b.loc[coefs_b.term=="tpoly4_lag"+str(l),"estimate"].values[0]
                    i1 = coefs_b.loc[coefs_b.term=="tmean:tpoly1_lag"+str(l),"estimate"].values[0]
                    i2 = coefs_b.loc[coefs_b.term=="tmean:tpoly2_lag"+str(l),"estimate"].values[0]
                    i3 = coefs_b.loc[coefs_b.term=="tmean:tpoly3_lag"+str(l),"estimate"].values[0]
                    i4 = coefs_b.loc[coefs_b.term=="tmean:tpoly4_lag"+str(l),"estimate"].values[0]

                    lograte_pred_cf = event_n["tcfpoly1_lag"+ll]*(c1 + i1*event_n["tmnew"]) + \
                                      event_n["tcfpoly2_lag"+ll]*(c2 + i2*event_n["tmnew"]) + \
                                      event_n["tcfpoly3_lag"+ll]*(c3 + i3*event_n["tmnew"]) + \
                                      event_n["tcfpoly4_lag"+ll]*(c4 + i4*event_n["tmnew"])
                    if no_event_counterfactual==False:
                        lograte_pred_clm = event_nbase["tcfpoly1_lag"+ll]*(c1 + i1*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly2_lag"+ll]*(c2 + i2*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly3_lag"+ll]*(c3 + i3*event_nbase["tmnew"]) + \
                                      event_nbase["tcfpoly4_lag"+ll]*(c4 + i4*event_nbase["tmnew"])
                    else:
                        lograte_pred_clm = event_n["tclmpoly1_lag"+ll]*(c1 + i1*event_n["tmnew"]) + \
                                       event_n["tclmpoly2_lag"+ll]*(c2 + i2*event_n["tmnew"]) + \
                                       event_n["tclmpoly3_lag"+ll]*(c3 + i3*event_n["tmnew"]) + \
                                       event_n["tclmpoly4_lag"+ll]*(c4 + i4*event_n["tmnew"])
                    
                mort_effect_pct = mort_effect_pct + (np.exp(lograte_pred_cf.values - lograte_pred_clm.values) - 1)
            event_n["mortality_change_pct"] = mort_effect_pct
            # we're multiplying by baseline deaths, which are zero for countries for which
            # we don't have mortality data, so they won't contribute to total heat mortality 
            event_n["temp_deaths"] = event_n.mortality_change_pct*event_n.n_deaths 
            #for q in quantiles:
            #    mortality.loc[:,:,g,q,i] = event_g.loc[event_g["quantile"]==q,["week","nuts","temp_deaths"]].pivot(index="nuts",columns="week",values="temp_deaths").values
            mortality.loc[:,:,g,n] = event_n[["week","nuts","temp_deaths"]].pivot(index="nuts",columns="week",values="temp_deaths").values
            if n==1:
                population.loc[nuts] = event_n[["week","nuts","population"]].pivot(index="nuts",columns="week",values="population").loc[nuts,:].values[:,0]
   

    mortality_out = mortality.sum(dim="nuts")
    mortality_out.name = "deaths"
    mortality_out.attrs["creation_date"] = str(datetime.datetime.now())
    mortality_out.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
    mortality_out.attrs["variable_description"] = "deaths from temperature"
    mortality_out.attrs["created_from"] = os.getcwd()+"/Counterfactual_Event_Mortality.py"
    
    fname_out = loc_events+e+"_tmean-interaction_counterfactual_mortality"+adapt_str+cfstr+"_europewide.nc"
    mortality_out.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)

    # average by region for easier analysis later
    mortality_region = mortality.mean(dim=["sample"])
    mortality_region_ds = xr.Dataset({"mortality":(["nuts","week","gmt"],mortality_region.values),
                                       "population":(["nuts"],population.values)},
                                coords={"nuts":(["nuts"],mortality_region.nuts.values),
                                        "week":(["week"],week),
                                        "boot":(["boot"],boot),
                                        "gmt":(["gmt"],gmts)})

    mortality_region_ds.attrs["creation_date"] = str(datetime.datetime.now())
    mortality_region_ds.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
    mortality_region_ds.attrs["variable_description"] = "average deaths from temperature by region"
    mortality_region_ds.attrs["created_from"] = os.getcwd()+"/Counterfactual_Event_Mortality.py"
    
    fname_out = loc_events+e+"_tmean-interaction_counterfactual_mortality"+adapt_str+cfstr+"_byregion.nc"
    mortality_region_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)


    # group by country for easier analysis later
    countries = xr.DataArray([x[0:2] for x in nuts],coords=[nuts],dims=["nuts"])
    mortality.coords["country"] = countries
    mortality_country = mortality.groupby("country").sum()
    population.coords["country"] = countries
    population_country = population.groupby("country").sum()
    countries_uq = population_country.country.values

    # wrap into ds and write out
    mortality_country_ds = xr.Dataset({"mortality":(["country","week","gmt","sample"],mortality_country.values),
                                       "population":(["country"],population_country.values)},
                                coords={"country":(["country"],countries_uq),
                                        "week":(["week"],week),
                                        "boot":(["boot"],boot),
                                        "sample":(["sample"],mc),
                                        "gmt":(["gmt"],gmts)})

    mortality_country_ds.attrs["creation_date"] = str(datetime.datetime.now())
    mortality_country_ds.attrs["created_by"] = "Christopher Callahan, christophercallahan@stanford.edu"
    mortality_country_ds.attrs["variable_description"] = "deaths from temperature by country"
    mortality_country_ds.attrs["created_from"] = os.getcwd()+"/Counterfactual_Event_Mortality.py"
    
    fname_out = loc_events+e+"_tmean-interaction_counterfactual_mortality"+adapt_str+cfstr+"_bycountry.nc"
    mortality_country_ds.to_netcdf(fname_out,mode="w")
    print(fname_out,flush=True)