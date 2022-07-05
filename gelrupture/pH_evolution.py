import os, os.path
import argparse
import numpy as np
from scipy.optimize import fsolve
import check

#GDL hydrolysis characteristic time
tau = 13e3
k = 1/tau
#GDL hydrolysis kinetic constant equilibirum constant
kgdl=7.7
#gluconic acid equilibrium constant
pkahgl = 3.86
kahgl = 10**(-pkahgl)
#caseinate equivalent weak base equilibrium constant
pI = 4.6
kacas=10**(-pI)


def EqCasGlBase(variables, g, c, b, x=0, z0=0, w0=0):
    """System of equations describing the equilibrium between caseinate, gluconic acid and formiate"""
    y,z,w =  variables
    h = g*x*y - c*(z-z0) - b*(w-w0)
    eqGl = kahgl*(1-y) - h*y
    eqCas = kacas*z - h*(1-z)
    eqBase = ka*w - h*(1-w)
    return [eqGl, eqCas, eqBase]


def thermo_CasGLBase(g, c, b, x=0, z0=0, w0=0):
    """Solve the thermodynamic equilibrium between caseinate and gluconic acid, gluconic acid and formiate to find their advances"""
    return fsolve(EqCasGlBase, [1, z0, w0], args=(g, c, b, x, z0, w0))

def EqCasGlBase_infty(variables, g, c, b, z0=0, w0=0):
    """System of equations describing the final equilibrium between caseinate, GDL and formiate"""
    x,y,z,w =  variables
    h = g*x*y - c*(z-z0) - b*(w-w0)
    eqGl = kahgl*(1-y) - h*y
    eqCas = kacas*z - h*(1-z)
    eqGDL = kgdl*(1-x) - x*(1-y)
    eqBase = ka*w - h*(1-w)
    return [eqGl, eqCas, eqGDL, eqBase]

def thermo_CasGLBase_infty(g, c, b, z0=0, w0=0):
    """Solve the thermodynamic equilibrium between caseinate and GDL to find their advances"""
    return fsolve(EqCasGlBase_infty, [1, 1, z0, w0], args=(g, c, b, z0, w0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the pH time evolution for GDL+casein(+weak base)')
    parser.add_argument('cas', type=float, help = 'weight percent of sodium caseinate')
    parser.add_argument('gdl', type=float, help = 'weight percent of GDL')
    parser.add_argument('--pH0', type=float, default=7., help='Initial pH')
    parser.add_argument('--base2cas', type=float, default=0, help='The molar ratio between weak base and caseinate (or gdl if cas==0).')
    parser.add_argument('--pKa', type=float, default='3.75', help = 'pKa of the weak base.')
    parser.add_argument('--no_clean_git',type=bool, default=False, help='Bypass checking wether my git repository is clean (Use only for tests).')

    args = parser.parse_args()

    if not args.no_clean_git:
        args.commit = check.clean_git(__file__).hexsha
    #args.datetime = datetime.now().isoformat()

    # Molar concentration of a weak base equivalent to 2%w of caséine.
    # Equivalent molar mass M=1020g/mol
    cas0=args.cas/100/1020*1000
    # Initial GDL concentration, in a 1/4 molar ratio with caseinate
    # M=178g/mol rho=1.7g/cm**3
    gdl0=args.gdl/100/178*1000
    #initial weak base concentration
    if args.cas>0:
        base0 = cas0 * args.base2cas
    else:
        base0 = gdl0 * args.base2cas

    #Knowing the initial pH, what is the ratio of casein in the commercial 'caseinate' powder?
    eqCas = lambda z: kacas*z - 10**(-args.pH0)*(1-z)
    z0, = fsolve(eqCas, 0)

    #Knowing the initial pH, what is the ratio of conjugate weak acid in the weak base solution?
    ka = 10**(-args.pKa)
    eqBase = lambda w: ka*w - 10**(-args.pH0)*(1-w)
    w0, = fsolve(eqBase, 0)

    #Solve for some advances of the hydrolysis between 0 and 1
    xs = np.unique(np.concatenate((np.linspace(0, 1), np.logspace(-np.log10(tau),0, 1000, base=10))))
    avancees = np.array([
        thermo_CasGLBase(gdl0, cas0, base0, x, z0, w0)
        for x in xs
    ])
    ys, zs, ws = avancees.T

    #H+ concentration
    h = gdl0*xs*ys - cas0*(zs-z0) - base0*(ws-w0)

    #Time dependence comes from the kinetics of GDL hydrolysis.
    dxdts = k*((1-xs) - xs*(1-ys)/kgdl)
    dxs = np.diff(xs)
    dts = dxs/dxdts[:-1]
    ts = np.concatenate(([0], np.cumsum(dts)))

    #final equilibirum
    xinf, yinf, zinf, winf = thermo_CasGLBase_infty(g=gdl0, c=cas0, b=base0, z0=z0, w0=w0)
    hinf = gdl0*xinf*yinf - cas0*(zinf-z0) - base0*(winf-w0)
    pHinf = -np.log10(hinf)
    print(f'final pH {pHinf:0.2f}')

    #truncate results at final equilibrium
    ixinf = np.nonzero(xs>xinf)[0][0]
    ts[ixinf] = np.inf
    ts = ts[:ixinf+1]
    pH = -np.log10(h)
    pH[ixinf] = pHinf
    pH = pH[:ixinf+1]

    outname = f'cas{args.cas:g}_GDL{args.gdl:g}'
    if args.base2cas>0:
        outname+=f'_base{args.base2cas:g}_pKa{args.pKa}'
    np.savetxt(
        outname+'.tpH',
        np.column_stack((ts, pH)),
        delimiter='\t',
        fmt=['%g', '%g'],
        header='t\tpH')
