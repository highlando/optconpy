#!/bin/bash
# use this script by calling `source get-upd-gh-deps.sh` in the console

mkdir dolfin_navier_scipy
cd dolfin_navier_scipy
rm *.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/data_output_utils.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/get_exp_nsmats.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/problem_setups.py
wget https://raw.githubusercontent.com/highlando/dolfin_navier_scipy/master/dolfin_navier_scipy/dolfin_to_sparrays.py
touch __init__.py
cd ..

mkdir sadptprj_riclyap_adi
cd sadptprj_riclyap_adi
rm lin_alg_utils.py proj_ric_utils.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/lin_alg_utils.py
wget https://raw.githubusercontent.com/highlando/sadptprj_riclyap_adi/master/proj_ric_utils.py
touch __init__.py
cd ..

mkdir distr_control_fenics
cd distr_control_fenics
rm cont_obs_utils.py
wget https://raw.githubusercontent.com/highlando/distr_control_fenics/master/cont_obs_utils.py
touch __init__.py
cd ..
