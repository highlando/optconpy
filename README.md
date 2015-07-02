# optconpy

Python suite for solving velocity tracking problems on finite horizones by the Riccati based ansatz described in my PhD thesis.

To get started, run

```
python run_optcont.py
```

There all parameters are set. You will have to create the folders 
```
mkdir data
mkdir results
```
in which the data and the results are stored.

The computed results are stored as *json* files that are input to the post processing and plotting routines from the submodule `plot_utils`.

## Dependencies:
 * Numpy 1.8.2
 * Scipy 0.13.2
 * dolfin 1.3.0 - the dolfin interface to [FEniCS](www.fenicsproject.org)
and my own python modules
 * [`dolfin_navier_scipy`](www.github.com/highlando/dolfin_navier_scipy)
 * [`sadptprj_riclyap_adi`](www.github.com/highlando/sadptprj_riclyap_adi)
 * [`distr_control_fenics`](www.github.com/highlando/distr_control_fenics)
