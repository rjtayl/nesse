# nesse
Nab Event Shape Simulation Effort (NESSE) is a python based solid state detector simulation developed for the Nab experiment. 


### Dependencies
Please insure the following dependencies are installed if installing nesse manually.

- Cython==3.1.2
- uproot==4.3.7
- numpy==1.24.3
- scipy
- tqdm
- matplotlib
- numba
- h5py
- pandas
- pyMSVC


### Install
nesse can either be installed using pip or manually.

To install with pip 

```
python3 -m pip install nesse
```

To install manually git clone and append the path to the nesse src directory like below:
```
import sys
sys.path.append(path_to_nesse+"/src/")
import nesse
```

Please see the tutorial notebook for more instruction on how to use nesse. 
