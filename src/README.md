# Battery sizing `source code`
Last updated by **Yi Ju** on **2022/08/24**

## urgent issues
write down any urgent issues here. suggested format:
- 2022/08/13: Yi Ju - **[the name of class]-[the name of method]-[line numbers]**: description of the problem. *(2022/08/13: [who help to solve this]: descriptions)*


## briefs
now, there are the following scripts under this folder:
- **core**
    - [`optimizer.py`](#optimizer)
- **support**
    - [`battery_model.py`](#battery_model)
    - [`data_loader.py`](#data_loader)
    - [`utils\utils.py`]

if you are working on a notebook (which should be placed under folder `\notebooks`) and wish to call some functions in above scripts, include the following codes in your notebook:

```
import sys
src_path = sys.path[0].replace("notebooks", "src")
if src_path not in sys.path:
    sys.path.append(src_path)
from battery_model import *
from optimizer import *
```

you should have `gurobi` installed before you can successfully run optimization models. For Anaconda users, use the following commands in the terminal:

```
conda activate [the enviornment you are to work with]
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi
```
About python envionment: [link](https://docs.conda.io/projects/conda/en/main/user-guide/tasks/manage-environments.html#activate-env). 
More about installing gurobi: [link](https://www.gurobi.com/documentation/9.5/quickstart_mac/cs_python_installation_opt.html). 
More on Gurobi: [link](https://www.gurobi.com/).

### When editting codes:
- write `# update 08/13 (Yi): ` to briefly summarize the changes you make, and add a log at the beginning of the script.
- use/search tag `TODO` or `FIXME` to mark those which haven't finished / may have problems.


## updates
on 2022/08/26, Yi made (is going to make) the following updates:
- DONE: deal with EV sessions whose td > t+K (in `Battery_optimizer.params_reg`)
- DONE: a module for loading data (bld, pv, ev) (in `data_loader.py`)
- TODO: a module for ev status management and logging
- TODO: simulation pipeline w/ MPC
- TODO: a general experimental trial manager

## <a name="optimizer">`optimizer.py`</a>
now, there are the following class under this script:
- `Battery_optimizer`:

### `Battery_optimizer`:
**public methods**:

- `__init__`
- `optimize_battery_size`
- `estimate_daily_TCO`
- `get_control_sequence`

## <a name="battery_model">`battery_model.py`</a>
now, there are the following class under this script:
- `Battery_base`:

### `Battery_base`:
**public methods**:

- `__init__`
- `update_soc`
- `should_renew`
- `copy_params`
- `set_capacity`, `set_params`, `get_params`, `get_states`
- `save_records`, `recover_records`

## <a name="data_loader">`data_loader.py`</a>
now, there are the following class/function under this script:
- `DataLoader`
    - `UCSD_dataloader`
- `ev_data_loader`

### `DataLoader`:
This is an abstract class providing some general methods that all successors may use. For a specific class, (usually) only need to override method `load_data_tmp`

**public methods**:

- `__init__`

*the following can be called directly, but not recommended / required*

- `load_data_tmp`
- `should_renew`
- `align_time_range`
- `rescale_load`



