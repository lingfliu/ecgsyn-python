
import math
import numpy as np
import scipy as sp
from scipy import  pi as PI
import scipy.integrate as integrate

J = sp.sqrt(-1)

'''
ecgsyn implementation
function [s, ipeaks] = ecgsyn(sfecg,N,Anoise,hrmean,hrstd,lfhfratio,sfint,ti,ai,bi)
sfecg: sampling freq ecg
N: heart beat number
anoise: additive noise
hrmean
hrstd
lfhfratio: LF/HF ratio
sfint: internal sampling frequency
ti: angles of extremea [-70 -15 0 15 100] degrees
ai = z-position of extrema [1.2 -5 30 -7.5 0.75]
bi = Gaussian width of peaks [0.25 0.1 0.1 0.1 0.4]

====================================================================
Original copyrights:

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

ecgsyn.m and its dependents are freely availble from Physionet - 
http://www.physionet.org/ - please report any bugs to the authors above.
====================================================================
'''
def ecgsyn(sfecg=256,
           N=256,
           anoise=0,
           hrmean=60,
           hrstd=1,
           lfhfratio=0.6,
           sfint=512,
           ti=[-70, -15, 0, 15, 100],
           ai=[1.2, -5, 30, -7.5, 0.75], # PQRST
           bi=[0.25, 0.1, 0.1, 0.1, 0.4] # PQRST
           ):
    # adjust extrema parameters for mean heart rate
    hrfact = math.sqrt(hrmean / 60)
    hrfact2 = math.sqrt(hrfact)
    bi = [b*hrfact for b in bi]

    ti = [tii*PI/180 for tii in ti]
    ti = [hrfact2*ti[0], hrfact*ti[1], ti[2], hrfact*ti[3], hrfact2*ti[4]]

    # check that sfint is an integer multiple of sfecg
    q = np.round(sfint/sfecg)
    qd = sfint//sfecg
    if not q == qd:
        print('sfint must be integer multiple of sfecg')
        return

    # define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and respiratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # calculate time scales for rr and total output
    sampfreqrr = 1
    trr = 1 / sampfreqrr
    tstep = 1 / sfecg
    rrmean = (60 / hrmean)
    Nrr = 2 ** (math.ceil(sp.log2(N * rrmean / trr)))

    # compute rr process
    rr0 = rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sampfreqrr, Nrr)

    # upsample rr time series from 1 Hz to sfint Hz
    idx = np.arange(0.0, len(rr0), 1.0)
    idx_interp = np.arange(0.0, len(rr0), 1/sfint)
    rr = np.interp(idx_interp, idx, rr0)

    # make the rr n time series
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg = 0

    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(np.round(tecg / dt))-1
        for ii in np.arange(i, ip):
            if ii >= len(rr):
                break
            rrn[ii] = rr[i]
        i = ip
    Nt = len(rr)-1

    x0 = [1, 0, 0.04]
    Tspan = np.arange(0, (Nt)*dt, dt)

    '''ode with dop853'''
    # solver = integrate.ode(desrivecgsyn)
    # solver.set_integrator('dop853')
    # solver.set_f_params([], rrn, sfint, ti, ai, bi)
    # t0 = 0.0
    # solver.set_initial_value(x0, t0)
    # X0 = np.empty((len(Tspan), 3))
    # X0[0] = x0
    # k = 1
    # while solver.successful() and solver.t < (Nt-1)*dt:
    #     solver.integrate(Tspan[k])
    #     X0[k] = solver.y
    #     k += 1

    X0 = sp.integrate.odeint(desrivecgsyn, x0, Tspan, args=([], rrn, sfint, ti, ai, bi))

    # downsample to required sfecg
    X = X0[::int(q)]

    # extract R - peaks times

    ipeaks = detectpeaks(X, ti, sfecg)

    # Scale signal to line between - 0.4 and 1.2 mV
    z = [xx[2] for xx in X]
    zmin = np.min(z)
    zmax = np.max(z)
    zrange = zmax - zmin
    z = (z - zmin) * (1.6) / zrange - 0.4

    # include additive uniformly distributed measurement noise
    eta = 2 * np.random.rand(len(z)) - 1
    s = z + anoise * eta

    return (s, ipeaks)

def rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    w1 = 2*PI*flo
    w2 = 2*PI*fhi
    c1 = 2*PI*flostd
    c2 = 2*PI*fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60/hrmean
    rrstd = 60*hrstd/(hrmean*hrmean)

    df = sfrr/n
    w = np.arange(0, n)*2*PI*df
    dw1 = w-w1
    dw2 = w-w2

    Hw1 = sig1*sp.exp(-0.5*(dw1/c1)**2)/sp.sqrt(2*PI*c1**2)
    Hw2 = sig2*sp.exp(-0.5*(dw2/c2)**2)/sp.sqrt(2*PI*c2**2)
    Hw = Hw1 + Hw2
    Hw0 = []
    for h in Hw[0:n//2]:
        Hw0.append(h)
    for h in Hw[n//2-1::-1]:
        Hw0.append(h)
    Sw = [sfrr/2*sp.sqrt(h) for h in Hw0]

    ph0 = 2*PI*sp.random.rand(n//2-1)
    ph = []
    ph.append(0)
    for v in ph0:
        ph.append(v)
    ph.append(0)
    for v in ph0[::-1]:
        ph.append(-v)
    SwC = [Sw[m]*sp.exp(J*ph[m]) for m in range(len(Sw))]
    x = (1/n)*sp.real(sp.ifft(SwC))

    xstd = sp.std(x)

    ratio = rrstd/xstd

    rr = rrmean + x*ratio

    return rr


def detectpeaks(X, thetap, sfecg):
    N = len(X)
    irpeaks = np.zeros(N)

    theta = [sp.arctan2(X[m][1], X[m][0]) for m in range(len(X))]
    ind0 = -1*np.ones(N)
    for i in range(N-1):
        j = [v for v in filter(lambda x:theta[i]<= thetap[x] and thetap[x] <= theta[i+1], range(len(thetap)))]
        if len(j) > 0:
            d1 = thetap[j[0]] - theta[i]
            d2 = theta[i + 1] - thetap[j[0]]

            if d1 < d2:
                ind0[i] = j[0]
            else:
                ind0[i+1] = j[0]

    d = sp.ceil(sfecg / 64)
    d = np.max([2, d])
    ind = -1*np.ones(N)
    z = [xx[2] for xx in X]
    zmin = np.min(z)
    zmax = np.max(z)
    zext = [zmin, zmax, zmin, zmax, zmin]
    sext = [1, -1, 1, -1, 1]
    for i in range(5):
        # clear ind1 Z k vmax imax iext
        ind1 = [v for v in filter(lambda a: ind0[a]==i, range(len(ind0)))]

        if len(ind1) == 0:
            continue
        n = len(ind1)

        Z = np.ones(shape=(n, int(2 * d + 1))) * zext[i] * sext[i]
        for j in np.arange(-d,d+1,1):
            k = [v for v in filter(lambda a: 0<= ind1[a]+j and ind1[a]+j <= N-1, range(len(ind1)))]
            for kk in k:
                Z[kk][int(d + j)] = z[int(ind1[kk] + j-1)] * sext[i]

        vmax = np.max(Z, axis=1)
        ivmax = np.argmax(Z, axis=1)
        iext = np.add(ivmax,ind1)
        iext = np.add(iext, -d-2)
        for ii in iext:
            ind[int(ii)] = i

    return ind


def desrivecgsyn(y0, t, flag, rr, sfint, ti, ai, bi):
# def desrivecgsyn(t, y0, flag, rr, sfint, ti, ai, bi):
    # function dxdt = derivsecgsyn(t, x, flag, rr, sfint, ti, ai, bi)
    # dxdt = derivsecgsyn(t, x, flag, rr, sampfreq, ti, ai, bi)
    # ODE file for generating the synthetic ECG
    # This file provides dxdt = F(t, x) taking input paramters:
    # rr: rr process
    # sfint: Internal sampling frequency[Hertz]
    # Order of extrema: [P Q R S T]
    # ti = angles of extrema[radians]
    # ai = z - position of extrema
    # bi = Gaussian width of peaks

    # Copyright(c) 2003 by Patrick McSharry & Gari Clifford, All Rights Reserved
    # See IEEE Transactions On Biomedical Engineering, 50(3), 289 - 294, March 2003.
    # Contact P.McSharry(patrick AT mcsharry DOT net) or % G.D.Clifford(gari AT mit DOT edu)

    xi = sp.cos(ti)
    yi = sp.sin(ti)
    ta = sp.arctan2(y0[1], y0[0])
    r0 = 1
    a0 = 1.0 - sp.sqrt(y0[0] ** 2 + y0[1]**2 ) / r0
    ip = int(sp.floor(t * sfint))
    w0 = 2 * PI / rr[ip]

    fresp = 0.25
    zbase = 0.005 * sp.sin(2 * PI * fresp * t)

    dx1dt = a0 * y0[0] - w0 * y0[1]
    dx2dt = a0 * y0[1] + w0 * y0[0]

    # dti = np.remainder(ta - ti, 2 * PI)
    dti = [ta-tii for tii in ti]
    for m in range(len(dti)):
        if dti[m] >= 0:
            dti[m] = np.remainder(dti[m], 2*PI)
        else:
            dti[m] = -np.remainder(-dti[m],2*PI)

    dx3dt = -np.sum([ai[m]*dti[m]*np.exp(-0.5*(dti[m]/bi[m])**2) for m in range(len(ai))]) - 1.0*(y0[2]-zbase)

    return [dx1dt, dx2dt, dx3dt]
