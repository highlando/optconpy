from optcont_main import optcon_nse

optcon_nse(N=15, Nts=30, nu=1e-2, clearprvveldata=True,
           closed_loop=False,
           ini_vel_stokes=True, stst_control=True, t0=0.0, tE=1.0)
