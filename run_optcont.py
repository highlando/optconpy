from optcont_main import optcon_nse

closed_loop = 1
stst_control = 0

optcon_nse(N=15, Nts=10, nu=1e-2, clearprvveldata=True,
           closed_loop=closed_loop, stst_control=stst_control,
           ini_vel_stokes=True, t0=0.0, tE=1.0)
