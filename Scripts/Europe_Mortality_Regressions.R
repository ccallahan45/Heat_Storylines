# temperature and mortality, europe-wide

rm(list=ls())
gc()

# libs
library(ggplot2) #v3.5.1
library(tidyverse) #v2.0.0
library(fixest) #v0.12.1
library(lubridate) #v1.9.4

filter <- dplyr::filter
select <- dplyr::select

## locations
folder <- "/your/path/to/project/folder/"
loc_panel <- paste0(folder,"Data/Panel/")
loc_reg <- paste0(folder,"Data/RegressionResults/")



## read data
data <- read_rds(paste0(loc_panel,"eurostat_eobs_weekly_mortality_temperature_2000-2022.rds")) %>% 
  arrange(nuts,nuts_level,country,name,year,week,age) 
## remember its nuts3 except for germany which is nuts1
## and germany only has total age not age-specific
#print(unique(data$countryname))

## lags
data <- data %>% arrange(nuts,nuts_level,country,name,year,week)
for (l in c(1:6)){
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tpoly1_lag",l,sep="") := dplyr::lag(tpoly1,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tpoly2_lag",l,sep="") := dplyr::lag(tpoly2,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tpoly3_lag",l,sep="") := dplyr::lag(tpoly3,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tpoly4_lag",l,sep="") := dplyr::lag(tpoly4,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("txpoly1_lag",l,sep="") := dplyr::lag(txpoly1,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("txpoly2_lag",l,sep="") := dplyr::lag(txpoly2,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("txpoly3_lag",l,sep="") := dplyr::lag(txpoly3,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("txpoly4_lag",l,sep="") := dplyr::lag(txpoly4,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tnpoly1_lag",l,sep="") := dplyr::lag(tnpoly1,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tnpoly2_lag",l,sep="") := dplyr::lag(tnpoly2,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tnpoly3_lag",l,sep="") := dplyr::lag(tnpoly3,l)) -> data
  data %>% group_by(nuts,age) %>% 
    mutate(!!paste("tnpoly4_lag",l,sep="") := dplyr::lag(tnpoly4,l)) -> data
}



## how many unique regions do we have for each age group?
data_n <- data %>% filter(year<=2019,year>=2015)
print(unique(data_n[which((data_n$age=="total")&(data_n$all_non_na_1519==1)&(data_n$any_na_1519==0)),"nuts"]))
# 924
print(unique(data_n[which((data_n$age=="65+")&(data_n$all_non_na_1519==1)&(data_n$any_na_1519==0)),"nuts"]))
# 908
print(unique(data_n[which((data_n$age=="0-64")&(data_n$all_non_na_1519==1)&(data_n$any_na_1519==0)),"nuts"]))
# 908


## interaction with mean temperature by age
# calc tmean
data_tmean <- data %>% filter(year>=2000,year<=2019) %>% 
  group_by(nuts,age) %>% mutate(tmean=mean(tm,na.rm=T))
# set years
y1 <- 2015
y2 <- 2019
# set nboot
nboot <- 500
# set nlag
nlag <- 3

ages <- c("total") #,"65+","0-64")
for (agegrp in ages){
  # set seed
  set.seed(100)
  print(agegrp)
  #if(age=="total"){nboot<-100}else{nboot<-100}
  # data 
  data_reg <- data_tmean %>% 
    filter(year>=y1,year<=y2,!is.na(rate),age==agegrp,all_non_na_1519==1,any_na_1519==0)  
  
  # formula
  form <- paste0("log(rate) ~ (tpoly1 + tpoly2 + tpoly3 + tpoly4)*tmean")
  for (l in c(1:nlag)){
    form <- paste0(form," + (tpoly1_lag",l," + tpoly2_lag",l," + tpoly3_lag",l," + tpoly4_lag",l,")*tmean")
  }
  # run regression 
  for (b in c(1:nboot)){
    print(b)
    unityr <- as.vector(unique(data_reg$nuts))
    unityr_boot <- sample(unityr,size=length(unityr),replace=T)
    df_boot <- sapply(unityr_boot, function(x) which(data_reg[,'nuts']==x))
    data_boot <- data_reg[as.vector(unlist(df_boot)),]
    
    mdl <- feols(as.formula(paste0(form," | nuts^year + nuts^week")),
                 data=data_boot,weights=~population,notes=F)
    
    coefs_mdl <- broom::tidy(mdl)
    coefs_mdl$boot <- b
    if (b==1){
      coefs <- coefs_mdl
    } else {
      coefs <- rbind(coefs,coefs_mdl)
    }
  }
  write.csv(coefs,file=paste0(loc_reg,"europe_weekly_mortality_coefficients_tmean-interact_",nlag,"lag_",agegrp,"age_",y1,"-",y2,".csv"))
}



## now maximum and minimum temperature
# calc tmean
data_tmean <- data %>% filter(year>=2000,year<=2019) %>% 
  group_by(nuts,age) %>% mutate(tmean=mean(tm,na.rm=T))
# set years
y1 <- 2015
y2 <- 2019
# set nboot
nboot <- 500
# set nlag
nlag <- 3
# ages
agegrp <- "total"

# vars
tvars <- c("tx","tn")

for (tv in tvars){
  # set seed
  set.seed(100)
  print(tv)
  data_reg <- data_tmean %>% 
    filter(year>=y1,year<=y2,!is.na(rate),age==agegrp,all_non_na_1519==1,any_na_1519==0)  
  
  # formula
  form <- paste0("log(rate) ~ (",tv,"poly1 + ",tv,"poly2 + ",tv,"poly3 + ",tv,"poly4)*tmean")
  for (l in c(1:nlag)){
    form <- paste0(form," + (",tv,"poly1_lag",l," + ",tv,"poly2_lag",l," + ",tv,"poly3_lag",l," + ",tv,"poly4_lag",l,")*tmean")
  }
  # run regression 
  for (b in c(1:nboot)){
    print(b)
    unityr <- as.vector(unique(data_reg$nuts))
    unityr_boot <- sample(unityr,size=length(unityr),replace=T)
    df_boot <- sapply(unityr_boot, function(x) which(data_reg[,'nuts']==x))
    data_boot <- data_reg[as.vector(unlist(df_boot)),]
    
    mdl <- feols(as.formula(paste0(form," | nuts^year + nuts^week")),
                 data=data_boot,weights=~population,notes=F)
    
    coefs_mdl <- broom::tidy(mdl)
    coefs_mdl$boot <- b
    if (b==1){
      coefs <- coefs_mdl
    } else {
      coefs <- rbind(coefs,coefs_mdl)
    }
  }
  write.csv(coefs,file=paste0(loc_reg,"europe_weekly_mortality_coefficients_tmean-interact_",tv,"_",nlag,"lag_",agegrp,"age_",y1,"-",y2,".csv"))
}


