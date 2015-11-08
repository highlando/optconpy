#!/bin/bash

# use this script by calling `source get-upd-gh-deps.sh` in the console

mkdir dolfin_navier_scipy
cd dolfin_navier_scipy
rm dolfin_to_sparrays.py data_output_utils.py problem_setups.py stokes_navier_utils.py
touch __init__.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_to_sparrays.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/data_output_utils.py 
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/problem_setups.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/stokes_navier_utils.py
cd ..

mkdir sadptprj_riclyap_adi
cd sadptprj_riclyap_adi
rm lin_alg_utils.py proj_ric_utils.py
touch __init__.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/lin_alg_utils.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/proj_ric_utils.py
cd ..

mkdir distr_control_fenics
cd distr_control_fenics/
rm cont_obs_utils.py 
touch __init__.py
wget https://raw.githubusercontent.com/highlando/distr_control_fenics/master/cont_obs_utils.py
cd ..

