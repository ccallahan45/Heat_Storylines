# Intensifying risk of mass human heat mortality if historical weather patterns recur

This repository provides code and data for "Intensifying risk of mass human heat mortality if historical weather patterns recur," currently in review. You can read the preprint [here](https://eartharxiv.org/repository/view/8375/).

### Organization

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of most the figures and numbers cited in the text. Some of the files for the climate model projections, raw observational data, and damage estimates are quite large, so they are not provided here. Details are below.

- **Scripts/**: Code required to reproduce the findings of our work is included in this folder. Scripts are written in Python and R. The script titled `Plot_Storyline_Figures.ipynb` produces the main text figures and most of the numbers cited in the text. 

- **Figures/**: This is where figures will be saved if you run the scripts.

### Data

Much of the intermediate data required to reproduce the final figures and numbers in the paper are provided in the various folders within the Data directory, including the overall panel dataset that includes district-level mortality and temperature. However, much of the initial/raw data is too large to be hosted here. However, they are all publicly available at various locations:

- The **E-OBS** station-based observations are available [here](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php#datafiles).

- The **Eurostat** mortality and population data are available [here](https://ec.europa.eu/eurostat/data/database?node_code=demomwk). Weekly mortality is denoted by the `demo_r_mweek3` code and population is denoted by the `demo_r_pjanaggr3` code. 

- **CMIP6** monthly temperature data (used for the adaptation analysis) is generally available from the Earth System Grid Federation (e.g., [here](https://aims2.llnl.gov/search/cmip6/)). Our analyis uses monthly temperature ("tas_Amon") from the historical and SSP3-7.0 scenarios.

### Scripts

Each script performs one component of the analysis.

- `Aggregate_EuropeWide_E-OBS_Data.py` takes the raw E-OBS input data and aggregates it to district-level (NUTS) boundaries at the daily and weekly level. This script was run remotely on an HPC system and may require significant memory.

- `Construct_Eurostat_Panel.R` combines the E-OBS district-level data with matching weekly mortality and temperature data and produces a final panel dataset spanning 2000-2022.

- `Europe_Mortality_Regressions.R` performs the regression analysis to derive the exposure-response functions.
  
- `Plot_Event_Timeseries.ipynb` plots and sets the time periods of each event.

- `Counterfactual_Events.py` combines the E-OBS data and neural network predictions to create counterfactual versions of each historical event at varying levels of global warming. This script was run remotely on an HPC system and may require significant memory.

- `CMIP6_EU_Scaling.py` and `CMIP6_Adaptation_Scaling.ipynb` aggregate CMIP6 projections to the EU district level and calculate the scaling between global mean temperature and district-level temperature (for use in the adaptation analysis).

- `Counterfactual_Event_Mortality.py` performs the final calculations of each event's mortality at varying levels of global warming. 
