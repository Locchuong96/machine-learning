**1/ install miniconda**

`bash Miniconda3-py38_22.11.1-1-Linux-x86_64.sh`

[ENTER] -> yes -> no

**2/ check miniconda version**

`bin/conda --version`

**3/ check miniconda information**

`bin/conda info`

**4/ init miniconda**

`bin/conda init`

**5/ check miniconda virtual enviroment**

`bin/conda env list`

**6/ create virtual enviroment**

`bin/conda create py38 python=3.8`

**6/ activate virtual environment**

`source activate <yourenv>` or `bin/conda activate <yourenv>`

**7/ activate virtual environment**

`source deactivate <yourenv>` or `bin/conda deactivate`

**8/ install new package in virtual enviroment**

`bin/conda install -n <yourenv> <package>` or `pip install <package>`

**9/ remove virtual enviroment**

`bin/conda remove -n <yourenv> -all`

**10/ uninstall conda**
`sudo rm -rf <your_miniconda>`

[download](https://repo.anaconda.com/miniconda/)
