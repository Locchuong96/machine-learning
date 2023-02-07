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

`bin/conda create --name <env> --file requirements.txt`

**8/ activate virtual environment**

`source activate <yourenv>` or `bin/conda activate <yourenv>`

**9/ activate virtual environment**

`source deactivate <yourenv>` or `bin/conda deactivate`

**10/ list installed packages in virtual enviroment**

`<your miniconda3>/envs/<your env>/bin/pip list` or `<your miniconda3>/envs/<your env>/bin/pip3 list` 

example: `myconda/envs/py38/bin/pip list`

**11/ install new package in virtual enviroment**

`bin/conda install -n <yourenv> <package>` or `pip install <package>`

install new packages follow requirements.txt 

`<your miniconda3>/envs/<your env>/bin/pip install -r requirements.txt` or `<your miniconda3>/envs/<your env>/bin/pip3 install -r requirements.txt`

**12/ export installed packages to requirements.txt**

`<your miniconda3>/envs/<your env>/bin/pip freeze > requirements.txt` or `<your miniconda3>/envs/<your env>/bin/pip3 freeze > requirements.txt`

**13/ run conda**

`sudo <path to miniconda3>/envs/<your env>/bin/python <path to your script>/script.py`

**14/ remove virtual enviroment**

`bin/conda remove -n <yourenv> --all`

**15/ uninstall conda**
`sudo rm -rf <your miniconda>`

**Note**

- related to `.profile`,`.bash_profile`,`bashrc`

- remove hidden files `.condarc`, `.conda`,`continuum` in `home` directory, press `Ctrl+H` to display

[download](https://repo.anaconda.com/miniconda/)

[official-document](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/index.html)

[conda install requirements](https://linuxhint.com/conda-install-requirements-txt/)

[conda manage packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)

[uninstall conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html#:~:text=The%20installer%20prompts%20%E2%80%9CDo%20you,%E2%80%9D%20We%20recommend%20%E2%80%9Cyes%E2%80%9D.&text=If%20you%20enter%20%E2%80%9Cno%E2%80%9D%2C,your%20shell%20scripts%20at%20all.)

[conda wheel file](https://docs.conda.io/projects/conda-build/en/3.23.x/user-guide/wheel-files.html)

[pypi](https://pypi.org/) Find, install and publish Python packages with the Python Package Index

[wheel package](https://realpython.com/python-wheels/#:~:text=whl%20file%20is%20essentially%20a,a%20type%20of%20built%20distribution.)
