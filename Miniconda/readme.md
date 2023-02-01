**1/ install miniconda**

`bash Miniconda3-py38_22.11.1-1-Linux-x86_64.sh`

[ENTER] -> yes -> no

**2/ check miniconda version**

`bin/conda --version`

**3/ check miniconda information**

`bin/conda info`

**4/ init miniconda**

`bin/conda init`

**5/ config miniconda**

`conda config --set auto_activate_base false`

**6/ check miniconda virtual enviroment**

`bin/conda env list`

**7/ create virtual enviroment**

`bin/conda create py38 python=3.8`

**8/ activate virtual environment**

`source activate <yourenv>` or `bin/conda activate <yourenv>`

**9/ activate virtual environment**

`source deactivate <yourenv>` or `bin/conda deactivate`

**10/ install new package in virtual enviroment**

`bin/conda install -n <package> <yourenv>` or `pip install <package>`

**11/ remove virtual enviroment**

`bin/conda remove -n <yourenv> --all` or 

**12/ uninstall conda**
`sudo rm -rf <your_miniconda>`

**Note**

related to `profile`,`bash_profile`,`bash_login`

[download](https://repo.anaconda.com/miniconda/)
