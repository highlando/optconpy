from optcont_main import optcon_nse
import dolfin

from dolfin_navier_scipy.data_output_utils import logtofile

logtofile('logfile_rev1')

closed_loop = 1
stst_control = 0

outernwtnstps = 1

nwtn_adi_dict = dict(adi_max_steps=300,
                     adi_newZ_reltol=1e-7,
                     nwtn_max_steps=20,
                     nwtn_upd_reltol=4e-8,
                     nwtn_upd_abstol=1e-7,
                     verbose=True,
                     # ms=[-15.0, -10.0, -5.0, -3.0, -2.0, -1.5, -1.3, -1.0],
                     ms=[-5.0, -3.0, -2.0, -1.5, -1.3, -1.1, -1.0],
                     full_upd_norm_check=False,
                     check_lyap_res=False)

alphau = 1e-11
gamma = 1e-1
# ystarstr = ['0', '0']
ystarstr = ['-0.1*sin(5*3.14*t)', '0.1*sin(5*3.14*t)']

ystar = [dolfin.Expression(ystarstr[0], t=0),
         dolfin.Expression(ystarstr[1], t=0)]

scaletest = 0.2*1e1
optcon_nse(N=25, Nts=64*scaletest, nu=0.5e-2, clearprvveldata=True,
           closed_loop=closed_loop, stst_control=stst_control,
           ini_vel_stokes=True, t0=0.0, tE=0.1*scaletest,
           outernwtnstps=outernwtnstps,
           linearized_nse=True,
           alphau=alphau,
           gamma=gamma,
           ystar=ystar,
           # stokes_flow=True,
           nwtn_adi_dict=nwtn_adi_dict)
