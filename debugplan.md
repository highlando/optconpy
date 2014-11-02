time step sizes
===============
 * check the overshoots maybe due to trap rule
 
time dep stokes
===============
 * done: initial vel value -- check or set to stokes
 * done: feedback/through computed for stokes flow
 * done: norm of fbft converge to norm of stst fbft
 * done: stokes flow with time dep feedback/through
 * done: feedback/feedthrough loaded rightly in solve_nse
 * done: signs of mtxtb and A - UV

DONE: start!!! steady state stokes
============================
 * done: check `umat_c` in snu.solve_nse

done: Driven Cavity
=============
 * done: changes not visible in output
 * done: need check the outputplotting

Outer Newton
============

in every iteration, check
 * done: save the data newton steps - with `N`
 * done: mtxb -- changes
 * done: Z -- changes
 * feedback matrix in the snu.solve_nse changes
 * feedback (matrix.v) changes
 * velocity changes
 * check implementation of feedback + updates
   * in ADI iteration
   * in/for solve_nse
	
Check for Linearized Prob (like in the diss):
=============================================

 * done: time dep vel lin point for solve nse (not there yet)...
 
