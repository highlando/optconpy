import dolfin
import json
import numpy as np
import os
import glob

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau
import sadptprj_riclyap_adi.proj_ric_utils as pru

import distr_control_fenics.cont_obs_utils as cou

import solve_dae_ric as sdr


dolfin.parameters.linear_algebra_backend = 'uBLAS'


class ContParams():
    """define the parameters of the control problem

    as there are
    - dimensions of in and output space
    - extensions of the subdomains of control and observation
    - weighting matrices (if None, then massmatrix)
    - desired output
    """
    def __init__(self, odcoo, ystar=None):
        if ystar is None:
            self.ystarx = dolfin.Expression('0.0', t=0)
            self.ystary = dolfin.Expression('0.0', t=0)
            # if t, then add t=0 to both comps !!1!!11
        else:
            self.ystarx = ystar[0]
            self.ystary = ystar[1]

        self.NU, self.NY = 4, 4

        self.R = None
        # regularization parameter
        self.alphau = 1e-6
        self.V = None
        self.W = None

        self.ymesh = dolfin.IntervalMesh(self.NY-1, odcoo['ymin'],
                                         odcoo['ymax'])
        self.Y = dolfin.FunctionSpace(self.ymesh, 'CG', 1)
        # TODO: pass Y to cou.get_output_operator
        # TODO: by now we tacitly assume that V, W = MyC.T My^-1 MyC
        # if contp.V is None:
        #     contp.V = My
        # if contp.W is None:
        #     contp.W = My

    def ystarvec(self, t=None):
        """return the current value of ystar

        as np array [ystar1
                     ystar2] """
        if t is None:
            try:
                self.ystarx.t, self.ystary.t = t, t
            except AttributeError:
                pass  # everything's cool - ystar does not dep on t
            else:
                raise Warning('You need provide a time for ystar')
        else:
            try:
                self.ystarx.t, self.ystary.t = t, t
            except AttributeError:
                if self.ystarx is None:
                    pass
                else:
                    raise UserWarning('no time dependency of ystar' +
                                      'the provided t is ignored')

        if self.ystarx is None and self.ystary is not None:
            ysy = dolfin.interpolate(self.ystary, self.Y)
            return np.atleast_2d(ysy.vector().array()).T

        elif self.ystary is None and self.ystarx is not None:
            ysx = dolfin.interpolate(self.ystarx, self.Y)
            return np.atleast_2d(ysx.vector().array()).T

        elif self.ystary is not None and self.ystarx is not None:
            ysx = dolfin.interpolate(self.ystarx, self.Y)
            ysy = dolfin.interpolate(self.ystary, self.Y)
            return np.vstack([np.atleast_2d(ysx.vector().array()).T,
                              np.atleast_2d(ysy.vector().array()).T])

        else:
            raise UserWarning('need provide at least one component of ystar')


def time_int_params(Nts, t0=0.0, tE=1.0):
    dt = (tE - t0) / Nts
    sqzmesh = True
    # squeeze the mesh for shorter intervals towards the
    # initial and terminal point, False for equidist
    tmesh = get_tint(t0, tE, Nts, sqzmesh)

    tip = dict(t0=t0,
               tE=tE,
               dt=dt,
               Nts=Nts,
               tmesh=tmesh,
               vfile=None,
               pfile=None,
               Residuals=[],
               ParaviewOutput=True,
               proutdir='results/',
               prfprfx='',
               # parameters for newton adi iteration
               nwtn_adi_dict=dict(
                   adi_max_steps=200,
                   adi_newZ_reltol=1e-8,
                   nwtn_max_steps=16,
                   nwtn_upd_reltol=5e-8,
                   nwtn_upd_abstol=1e-7,
                   verbose=True,
                   full_upd_norm_check=False,
                   check_lyap_res=False
               ),
               compress_z=True,  # whether or not to compress Z
               comprz_maxc=50,  # compression of the columns of Z by QR
               comprz_thresh=5e-5,  # threshold for trunc of SVD
               save_full_z=False  # whether or not to save the uncompressed Z
               )

    return tip


def get_tint(t0, tE, Nts, sqzmesh):
    """set up the time mesh """
    if sqzmesh:
        taux = np.linspace(-0.5*np.pi, 0.5*np.pi, Nts+1)
        taux = (np.sin(taux) + 1)*0.5  # squeeze and adjust to [0, 1]
        tint = (t0 + (tE-t0)*taux).flatten()  # adjust to [t0, tE]
    else:
        tint = np.linspace(t0, tE, Nts+1).flatten()

    return tint


def get_datastr(time=None, meshp=None, nu=None, Nts=None,
                data_prfx='', **kw):

    return (data_prfx + 'time{1}_nu{2}_mesh{3}_Nts{4}').format(
        None, time, nu, meshp, Nts)


def save_output_json(ycomp, tmesh, ystar=None, fstring=None):
    """save the signals to json for postprocessing"""
    if fstring is None:
        fstring = 'nonspecified_output'

    jsfile = open(fstring, mode='w')
    jsfile.write(json.dumps(dict(ycomp=ycomp,
                                 tmesh=tmesh,
                                 ystar=ystar)))

    print 'output saved to ' + fstring
    print '\n to plot run the commands \n'
    print 'import plot_output as plo'
    print 'import optcont_main as ocm'
    print 'jsf = ocm.load_json_dicts("' + fstring + '")'
    print 'plo.plot_optcont_json(jsf, fname="' + fstring + '")\n'


def load_json_dicts(StrToJs):

    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    return JsDict


def get_convmats_rhs(strtodata, invinds=None, V=None, diribcs=None, **kw):
    """retrieving the time varying coefficient matrix and associated rhs,


    where the latter also contains the dirichlet bcs
    """

    prev_v = dou.load_npa(strtodata)

    (convc_mat, rhs_con,
     rhsv_conbc) = snu.get_v_conv_conts(prev_v=prev_v, invinds=invinds,
                                        V=V, diribcs=diribcs)

    return convc_mat, rhsv_conbc + rhs_con


def init_nwtnstps_value_dict(tmesh=None, data_prfx=None):
    cnd = {}
    for t in tmesh:
        cnd.update({t: {'v': data_prfx + '__cns_v_t{0}'.format(t),
                        'mtxtb': data_prfx + '__cns_mtxtb_t{0}'.format(t),
                        'w': data_prfx + '__cns_w_t{0}'.format(t)}})
        for fname in glob.glob(data_prfx + '__cns_*_t*'):
            os.remove(fname)

    return cnd


def optcon_nse(problemname='drivencavity',
               N=10, Nts=10, nu=1e-2, clearprvveldata=False,
               ini_vel_stokes=False, stst_control=False,
               closed_loop=True,
               outernwtnstps=1,
               t0=None, tE=None,
               use_ric_ini_nu=None, alphau=None,
               spec_tip_dict=None,
               nwtn_adi_dict=None,
               linearizednse=False,
               ystar=None):

    tip = time_int_params(Nts, t0=t0, tE=tE)
    if spec_tip_dict is not None:
        tip.update(spec_tip_dict)
    if nwtn_adi_dict is not None:
        tip['nwtn_adi_dict'] = nwtn_adi_dict

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)

    problemfem = problemdict[problemname]
    femp = problemfem(N)

    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    if linearizednse and not outernwtnstps == 1:
        raise Warning('Linearized problem can have only one Newton step')

    if closed_loop:
        if stst_control:
            data_prfx = ddir + 'stst_' + problemname + '__'
        else:
            data_prfx = ddir + 'tdst_' + problemname + '__'

    else:
        data_prfx = ddir + problemname + '__'

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        uspacedep = 0
    elif problemname == 'cylinderwake':
        uspacedep = 1

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'], nu)
    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure
    stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
    stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
    rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]

    # reduce the matrices by resolving the BCs
    (stokesmatsc, rhsd_stbc,
     invinds, bcinds, bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                                          femp['diribcs'])

    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds, 'bcvals': bcvals, 'invinds': invinds}
    femp.update(bcdata)

    # casting some parameters
    NV = len(femp['invinds'])

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   trange=tip['tmesh'],
                   get_datastring=get_datastr,
                   data_prfx=data_prfx,
                   clearprvdata=clearprvveldata,
                   paraviewoutput=tip['ParaviewOutput'],
                   vfileprfx=tip['proutdir']+'vel_',
                   pfileprfx=tip['proutdir']+'p_')

#
# Prepare for control
#

    contp = ContParams(femp['odcoo'], ystar)
    # casting some parameters
    NY, NU = contp.NY, contp.NU
    if alphau is not None:
        contp.alphau = alphau

    contsetupstr = problemname + '__NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=femp['cdcoo'], V=femp['V'],
                                          NU=NU, xcomp=uspacedep)
        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], NY=NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]

    # for further use:
    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')

    if contp.ystarx is None:
        c_mat = c_mat[NY:, :][:, :]  # TODO: Do this right
        mc_mat = mc_mat[NY:, :][:, :]  # TODO: Do this right
        y_masmat = y_masmat[:NY, :][:, :NY]  # TODO: Do this right

    mct_mat_reg = lau.app_prj_via_sadpnt(amat=stokesmatsc['M'],
                                         jmat=stokesmatsc['J'],
                                         rhsv=mc_mat.T,
                                         transposedprj=True)

    # set the weighing matrices
    contp.R = contp.alphau * u_masmat

#
# solve the differential-alg. Riccati eqn for the feedback gain X
# via computing factors Z, such that X = -Z*Z.T
#
# at the same time we solve for the affine-linear correction w
#

    # tilde B = BR^{-1/2}
    tb_mat = lau.apply_invsqrt_fromright(contp.R, b_mat, output='sparse')
    # tb_dense = np.array(tb_mat.todense())

    trct_mat = lau.apply_invsqrt_fromright(y_masmat,
                                           mct_mat_reg, output='dense')

    if closed_loop:
        cntpstr = 'NV{3}NY{0}NU{1}alphau{2}'.\
            format(contp.NU, contp.NY, contp.alphau, NV)
    else:
        cntpstr = ''

    # we gonna use this quite often
    M, A = stokesmatsc['M'], stokesmatsc['A']
    datastrdict = dict(time=None, meshp=N, nu=nu, Nts=Nts, data_prfx=data_prfx)

    # compute the uncontrolled steady state (Navier-)Stokes solution
    # as initial value
    if ini_vel_stokes:
        # compute the uncontrolled steady state Stokes solution
        ini_vel, newtonnorms = snu.solve_steadystate_nse(vel_nwtn_stps=0,
                                                         vel_pcrd_stps=0,
                                                         **soldict)
        soldict.update(dict(iniv=ini_vel))
    else:
        ini_vel, newtonnorms = snu.solve_steadystate_nse(**soldict)
        soldict.update(dict(iniv=ini_vel))

    if closed_loop:
        if stst_control:
            lin_point, newtonnorms = snu.solve_steadystate_nse(**soldict)
            # infinite control horizon, steady target state
            cdatstr = get_datastr(time=None, meshp=N, nu=nu,
                                  Nts=None, data_prfx=data_prfx)

            (convc_mat, rhs_con,
             rhsv_conbc) = snu.get_v_conv_conts(prev_v=0*lin_point,
                                                invinds=invinds,
                                                V=femp['V'],
                                                diribcs=femp['diribcs'])

            try:
                Z = dou.load_npa(cdatstr + cntpstr + '__Z')
                print 'loaded ' + cdatstr + cntpstr + '__Z'
            except IOError:
                if use_ric_ini_nu is not None:
                    cdatstr = get_datastr(nwtn=None, time=None, meshp=N,
                                          nu=use_ric_ini_nu, Nts=None,
                                          data_prfx=data_prfx)
                    try:
                        zini = dou.load_npa(ddir + cdatstr
                                            + cntpstr + '__Z')
                        print 'Initialize Newton ADI by Z from ' + cdatstr
                    except IOError:
                        raise Warning('No data for initialization of '
                                      ' Newton ADI -- need ' + cdatstr
                                      + '__Z')
                    cdatstr = get_datastr(meshp=N, nu=nu,
                                          data_prfx=data_prfx)
                else:
                    zini = None

                parnadi = pru.proj_alg_ric_newtonadi
                Z = parnadi(mmat=M, amat=-A-convc_mat,
                            jmat=stokesmatsc['J'],
                            bmat=tb_mat, wmat=trct_mat, z0=zini,
                            nwtn_adi_dict=tip['nwtn_adi_dict'])['zfac']

                dou.save_npa(Z, fstring=cdatstr + cntpstr + '__Z')
                print 'saved ' + cdatstr + cntpstr + '__Z'

                if tip['compress_z']:
                    Zc = pru.compress_Zsvd(Z, thresh=tip['comprz_thresh'],
                                           k=tip['comprz_maxc'])
                    Z = Zc

            fvnstst = rhs_con + rhsv_conbc + rhsd_stbc['fv'] +\
                rhsd_vfrc['fvc']

            # X = -ZZ.T
            mtxtb_stst = -pru.get_mTzzTtb(M.T, Z, tb_mat)
            mtxfv_stst = -pru.get_mTzzTtb(M.T, Z, fvnstst)

            fl = mc_mat.T * contp.ystarvec(0)

            wft = lau.solve_sadpnt_smw(amat=A.T+convc_mat.T,
                                       jmat=stokesmatsc['J'],
                                       rhsv=fl+mtxfv_stst,
                                       umat=mtxtb_stst,
                                       vmat=tb_mat.T)[:NV]
            # next_w = wft  # to be consistent with unsteady state

            auxstrg = cdatstr + cntpstr
            dou.save_npa(wft, fstring=cdatstr + cntpstr + '__w')
            dou.save_npa(mtxtb_stst, fstring=cdatstr + cntpstr + '__mtxtb')
            feedbackthroughdict = {None:
                                   dict(w=auxstrg + '__w',
                                        mtxtb=auxstrg + '__mtxtb')}

            cns = 0
            soldict.update(data_prfx=data_prfx+'_cns{0}'.format(cns))
            dictofvels = snu.\
                solve_nse(return_dictofvelstrs=True,
                          closed_loop=True,
                          static_feedback=True,
                          tb_mat=tb_mat,
                          lin_vel_point={None: 0*lin_point},
                          vel_pcrd_stps=0,
                          vel_nwtn_stps=1,
                          feedbackthroughdict=feedbackthroughdict, **soldict)

        else:  # time dep closed loop

            cns_data_prfx = 'data/cnsvars'
            invd = init_nwtnstps_value_dict
            curnwtnsdict = invd(tmesh=tip['tmesh'],
                                data_prfx=cns_data_prfx)
            # initialization: compute the forward solution
            dictofvels = snu.solve_nse(return_dictofvelstrs=True,
                                       **soldict)

            # function for the time depending parts
            # -- to be passed to the solver
            def get_tdpart(time=None, dictofvalues=None, feedback=False,
                           V=None, invinds=None, diribcs=None, **kw):

                curvel = dou.load_npa(dictofvalues[time])
                convc_mat, rhs_con, rhsv_conbc = \
                    snu.get_v_conv_conts(prev_v=curvel, invinds=invinds,
                                         V=V, diribcs=diribcs)
                return convc_mat, rhsv_conbc+rhs_con

            gttdprtargs = dict(dictofvalues=dictofvels,
                               V=femp['V'],
                               diribcs=femp['diribcs'],
                               invinds=invinds)

            # old version rhs
            # ftilde = rhs_con + rhsv_conbc + rhsd_stbc['fv']
            for cns in range(outernwtnstps):

                datastrdict.update(data_prfx=data_prfx+'_cns{0}'.
                                   format(cns))
                soldict.update(data_prfx=data_prfx+'_cns{0}'.
                               format(cns))

                sfd = sdr.solve_flow_daeric
                feedbackthroughdict = \
                    sfd(mmat=M, amat=A, jmat=stokesmatsc['J'],
                        bmat=b_mat,
                        # cmat=ct_mat_reg.T,
                        mcmat=mct_mat_reg.T,
                        v_is_my=True, rmat=contp.alphau*u_masmat,
                        vmat=y_masmat, rhsv=rhsd_stbc['fv'], rhsp=None,
                        tmesh=tip['tmesh'], ystarvec=contp.ystarvec,
                        nwtn_adi_dict=tip['nwtn_adi_dict'],
                        comprz_thresh=tip['comprz_thresh'],
                        comprz_maxc=tip['comprz_maxc'], save_full_z=False,
                        get_tdpart=get_tdpart, gttdprtargs=gttdprtargs,
                        curnwtnsdict=curnwtnsdict,
                        get_datastr=get_datastr, gtdtstrargs=datastrdict)

                for t in tip['tmesh']:  # feedbackthroughdict.keys():
                    curw = dou.load_npa(feedbackthroughdict[t]['mtxtb'])
                    print cns, t, np.linalg.norm(curw)

                cdatstr = get_datastr(time='all', meshp=N, nu=nu,
                                      Nts=None, data_prfx=data_prfx)

                if linearizednse:
                    dictofvels = snu.\
                        solve_nse(return_dictofvelstrs=True,
                                  closed_loop=True, tb_mat=tb_mat,
                                  lin_vel_point=dictofvels,
                                  feedbackthroughdict=feedbackthroughdict,
                                  vel_nwtn_stps=1,
                                  vel_pcrd_stps=0,
                                  **soldict)
                else:
                    dictofvels = snu.\
                        solve_nse(return_dictofvelstrs=True,
                                  closed_loop=True, tb_mat=tb_mat,
                                  feedbackthroughdict=feedbackthroughdict,
                                  vel_nwtn_stps=5,
                                  **soldict)

                # for t in dictofvels.keys():
                #     curw = dou.load_npa(dictofvels[t])
                #     print cns, t, np.linalg.norm(curw)

                gttdprtargs.update(dictofvalues=dictofvels)
    else:
        # no control
        feedbackthroughdict = None
        tb_mat = None
        cdatstr = get_datastr(meshp=N, nu=nu, time='all',
                              Nts=Nts, data_prfx=data_prfx)

        soldict.update(clearprvdata=True)
        dictofvels = snu.solve_nse(feedbackthroughdict=feedbackthroughdict,
                                   tb_mat=tb_mat, closed_loop=closed_loop,
                                   return_dictofvelstrs=True,
                                   static_feedback=stst_control,
                                   **soldict)

    (yscomplist,
     ystarlist) = dou.extract_output(dictofpaths=dictofvels,
                                     tmesh=tip['tmesh'],
                                     c_mat=c_mat, ystarvec=contp.ystarvec)

    save_output_json(yscomplist, tip['tmesh'].tolist(), ystar=ystarlist,
                     fstring=cdatstr + cntpstr + '__sigout')

    print 'dim of v :', femp['V'].dim()
    charlene = .15 if problemname == 'cylinderwake' else 1.0
    print 'Re = charL / nu = {0}'.format(charlene/nu)

if __name__ == '__main__':
    optcon_nse(N=12, Nts=10, nu=1e-2,  # clearprvveldata=True,
               ini_vel_stokes=True, stst_control=True, t0=0.0, tE=1.0)
    # optcon_nse(problemname='cylinderwake', N=3, nu=1e-3,
    #            clearprvveldata=False,
    #            t0=0.0, tE=1.0, Nts=25, stst_control=True,
    #            comp_unco_out=False,
    #            ini_vel_stokes=True, use_ric_ini_nu=None, alphau=1e-4)
