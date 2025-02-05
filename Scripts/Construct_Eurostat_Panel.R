# assembling eurostat weekly temperature and mortality panel

rm(list=ls())
gc()

# libs
library(ggplot2) #v3.5.1
library(tidyverse) #v2.0.0
library(fixest) #v0.12.1
library(sf) #v1.0-19

filter <- dplyr::filter
select <- dplyr::select

## locations
folder <- "/your/path/to/project/folder/"
loc_panel <- paste0(folder,"Data/Panel/")
loc_eobs <- paste0(folder,"Data/E-OBS/")
loc_eurostat <- paste0(folder,"Data/Eurostat/")

## read mortality data
mort_in <- read.csv(paste0(loc_eurostat,"demo_r_mweek3_linear.csv"))
# in the population data we only have greater than <15, 15-64, and 65+
# so let's just run 0-64 and 65+, and total 
mort <- mort_in %>% filter(age!="UNK",sex=="T") %>%
  mutate(age_group=case_when(age=="TOTAL" ~ 1,
                             !age%in%c("TOTAL","UNK","Y65-69","Y70-74","Y75-79","Y80-84","Y85-89","Y_GE90")~2,
                             age%in%c("Y65-69","Y70-74","Y75-79","Y80-84","Y85-89","Y_GE90")~3)) %>% 
  select(age_group,geo,TIME_PERIOD,OBS_VALUE) %>%
  group_by(geo,TIME_PERIOD,age_group,.drop=FALSE) %>% summarize(n_deaths=sum(OBS_VALUE,na.rm=T)) %>%
  mutate(age=case_when(age_group==1 ~ "total",age_group==2 ~ "0-64",age_group==3 ~ "65+")) %>%
  select(-age_group) %>% rename(nuts3=geo) %>%
  mutate(year=as.numeric(substr(TIME_PERIOD,1,4)),week=as.numeric(substr(TIME_PERIOD,7,8))) 

# we have age-specific numbers here
# obs flag of "p" means provisional data, we'll go with it for now

## read population data
pop <- read.csv(paste0(loc_eurostat,"demo_r_pjanaggr3_linear.csv")) %>%
  filter(sex=="T") %>% select(geo,TIME_PERIOD,age,OBS_VALUE) %>% 
  rename(nuts3=geo,year=TIME_PERIOD,population=OBS_VALUE)
pop_age <- pop %>% filter(age!="UNK") %>%
  mutate(age_group=case_when(age=="TOTAL" ~ 1,
                             !age%in%c("Y_GE65","TOTAL") ~ 2,
                             age == "Y_GE65" ~ 3)) %>%
  select(age_group,nuts3,year,population) %>% 
  group_by(nuts3,year,age_group,.drop=FALSE) %>% summarize(population=sum(population)) %>%
  mutate(age=case_when(age_group==1 ~ "total",age_group==2 ~ "0-64",age_group==3 ~ "65+")) %>% 
  select(-age_group)
## there are some regions in germany that have units in the pop data but not mortality
## and the pop data appear to have turkey data but the mortality data does not

# noting that we're calling the nuts column "nuts3" here but its actually multiple nuts levels
# we'll limit to nuts3 only later (and nuts1 in germany) 

print(head(pop_age))
print(head(mort))

## merge 
mort_pop <- mort %>% left_join(pop_age,by=c("nuts3","year","age")) %>% 
  mutate(rate=n_deaths/(population/1e5))

mort_pop %>% filter(nuts3%in%c("AL","AT","BE","CH","DE","FR","IT")) %>%
  filter(age=="total") %>% group_by(nuts3,year) %>%
  summarize(deaths=sum(n_deaths,na.rm=T),population=first(population)) %>%
  mutate(rate=deaths/(population/1e3)) %>% group_by(year) %>%
  summarize(mean_rate=weighted.mean(rate,population,na.rm=T)) -> df_test
plot(df_test$year,df_test$mean_rate)

## get e-obs data and match
eobs <- read.csv(paste0(loc_eobs,"e-obs_nuts3_weekly_temperature_popweight_1980-2023.csv")) %>%
  select(-X)
print(eobs)

## some countries have only higher-level units for mortality, e.g. germany
# so its nuts 3 except for germany which is nuts 1
# and germany only has total age not age-specific
head(eobs[eobs$LEVL_CODE==3,])
eobs_merge <- eobs %>% filter((LEVL_CODE==3) | (LEVL_CODE==1)&(CNTR_CODE=="DE")) %>% 
  rename(nuts3=NUTS_ID) %>% select(-year.week)

## merge
data <- eobs_merge %>% left_join(mort_pop,by=c("nuts3","year","week")) %>%
  rename(nuts=nuts3,nuts_level=LEVL_CODE,country=CNTR_CODE,yearweek=TIME_PERIOD,name=NUTS_NAME)
dim(data)
dim(data %>% filter(!is.na(rate)))

## which nuts regions have continuous coverage over 2015 to 2019?
data_na <- data %>% filter(year>=2015,year<=2019) %>% group_by(nuts,age) %>% 
       mutate(all_non_na_15_19 = as.numeric(all(!is.na(rate))),
              any_na_15_19 = as.numeric(any(is.na(rate)))) %>% 
       select(-c(nuts_level,yearweek))

## write out
## limit to 2022 because of lacking data on later years from either
## eurostat or e-obs
## 2022 is last full year where we have all the data
data_out <- data %>% filter(year>=2000,year<=2022) %>% 
  left_join(data_na_merge,by="nuts")
write_rds(data_out,paste0(loc_panel,"eurostat_eobs_weekly_mortality_temperature_2000-2022.rds"))


### other countries left out of the final data
##### ireland (IE) -- only has country-level mortality
##### croatia (HR) -- only has country-level mortality
##### north macedonia (MK) -- doesn't have mortality data
##### malta (MT) -- doesn't have climate data (likely too small/not overlapping E-OBS)
##### slovenia (SI) -- only has country-level mortality
##### turkey (TR) -- doesn't have mortality data



