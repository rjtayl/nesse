# nesse
Nab Event Shape Simulation Effort (NESSE) is a python based solid state detector simulation developed for the Nab experiment. 


### Dependencies
Please insure the following dependencies are installed if installing nesse manually.

- Cython
- uproot==4.3.7
- numpy==1.24.3
- scipy
- tqdm
- matplotlib
- numba
- h5py
- pandas
- pyMSVC

Currently this will only work for Python versions 3.7-3.11 due to some of the above dependencies. Python packaging does not like when you specify upper bounds on python versions so you, the user, will have check this yourself.
Warning: as of 0.2.0 it appears different systems require using different versions of cython, if you get a compilation error please try a different version of cython. 

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
