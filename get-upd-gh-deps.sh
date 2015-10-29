#!/bin/bash

# use this script by calling `source get-upd-gh-deps.sh` in the console

mkdir dolfin_navier_scipy
touch dolfin_navier_scipy/__init__.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_to_sparrays.py dolfin_navier_scipy/dolfin_to_sparrays.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/data_output_utils.py dolfin_navier_scipy/data_output_utils.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/data_output_utils.py dolfin_navier_scipy/stokes_navier_utils.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/data_output_utils.py dolfin_navier_scipy/problem_setups.py

mkdir sadptprj_riclyap_adi
touch sadptprj_riclyap_adi/__init__.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/lin_alg_utils.py sadptprj_riclyap_adi/lin_alg_utils.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/lin_alg_utils.py sadptprj_riclyap_adi/proj_ric_utils.py

mkdir distr_control_fenics
touch distr_control_fenics/__init__.py
wget https://raw.githubusercontent.com/highlando/distr_control_fenics/master/cont_obs_utils.py distr_control_fenics/cont_obs_utils.py

