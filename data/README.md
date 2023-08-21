# data
This folder is used to contain all raw data for potential use.
- Please do not manipulate raw data directly (e.g., edit data in the csv file); instead, write python script so that others can easily understand and reproduce your operations on the data.
    - If you edit the raw data and the edited data is served as data source for some experiment, then save the edited data in a new folder under `data`, together with the editing script (`.py`).
    - However, save the results of experiments somewhere under the folder `output` (07/30: hasn't created). 
- use the following codes to navigate to the data folder:
```
import sys
import os
# suppose this script is under the folder "src"
data_path = sys.path[0].replace("src", "data")
folder = "UCSD_raw_data"
fn = "....csv"
fn_path = os.path.join(data_path, folder, fn)
df = pd.read_csv(fn_path, ...)
```

## UCSD data
find UCSD data in folder `UCSD_raw_data`, which includes:
- PV harvest, with filename prefix "PV_"
- Building load, with prefix "BLD_"
- EV sessions, with prefix "EV_"
- Day-ahead time-of-use tariff of SDG&E (xxx), "Price_SDGE"
- Weather files that may help forecast load, "Weather_SanDiego"
- (pending) Greenhouse gas (GHG) factor of the grid 
- Detailed description of UCSD data, see [this paper](https://aip.scitation.org/doi/10.1063/5.0038650)