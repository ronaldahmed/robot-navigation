## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

"""
 Python module for computing Logistic Regression.
 
 Requires numarray or Numeric.

 Version: 20050711
 
 Contact:  Jeffrey Whitaker <jeffrey.s.whitaker@noaa.gov>
 
 This code is released into the Public Domain as is.
 No support or warrantee is provided. Comments, bug reports
 and enhancements are welcome.
"""

_use_numarray=False
try:
    import numarray as NA
    import numarray.linear_algebra as LA
    _use_numarray = True
except:
    try:
        import numpy.oldnumeric as NA
        import numpy.oldnumeric.linear_algebra as LA
    except:
        raise ImportError, 'Requires numarray or Numeric'

def _simple_logistic_regression(x,y,beta_start=None,verbose=False,
                               CONV_THRESH=1.e-3,MAXIT=500):
    """
 Faster than logistic_regression when there is only one predictor.
    """
    if len(x) != len(y):
        raise ValueError, "x and y should be the same length!"
    if beta_start is None:
        beta_start = NA.zeros(2,x.dtype.char)
    iter = 0; diff = 1.; beta = beta_start  # initial values
    if verbose:
        print 'iteration  beta log-likliehood |beta-beta_old|' 
    while iter < MAXIT:
        beta_old = beta 
        p = NA.exp(beta[0]+beta[1]*x)/(1.+NA.exp(beta[0]+beta[1]*x))
        l = NA.sum(y*NA.log(p) + (1.-y)*NA.log(1.-p)) # log-likliehood
        s = NA.array([NA.sum(y-p), NA.sum((y-p)*x)])  # scoring function
        # information matrix
        J_bar = NA.array([[NA.sum(p*(1-p)),NA.sum(p*(1-p)*x)],
                          [NA.sum(p*(1-p)*x),NA.sum(p*(1-p)*x*x)]])
        beta = beta_old + NA.dot(LA.inverse(J_bar),s) # new value of beta
        diff = NA.sum(NA.fabs(beta-beta_old)) # sum of absolute differences
        if verbose:
            print iter+1, beta, l, diff
        if diff <= CONV_THRESH: break
        iter = iter + 1
    return beta, J_bar, l

def logistic_regression(x,y,beta_start=None,verbose=False,CONV_THRESH=1.e-3,
                        MAXIT=500):
    """
 Uses the Newton-Raphson algorithm to calculate a maximum
 likelihood estimate logistic regression.
 The algorithm is known as 'iteratively re-weighted least squares', or IRLS.

 x - rank-1 or rank-2 array of predictors. If x is rank-2,
     the number of predictors = x.shape[0] = N.  If x is rank-1,
     it is assumed N=1.
     
 y - binary outcomes (if N>1 len(y) = x.shape[1], if N=1 len(y) = len(x))
 
 beta_start - initial beta vector (default zeros(N+1,x.dtype.char))
 
 if verbose=True, diagnostics printed for each iteration (default False).
 
 MAXIT - max number of iterations (default 500)
 
 CONV_THRESH - convergence threshold (sum of absolute differences
  of beta-beta_old, default 0.001)

 returns beta (the logistic regression coefficients, an N+1 element vector),
 J_bar (the (N+1)x(N+1) information matrix), and l (the log-likeliehood).
 
 J_bar can be used to estimate the covariance matrix and the standard
 error for beta.
 
 l can be used for a chi-squared significance test.

 covmat = inverse(J_bar)     --> covariance matrix of coefficents (beta)
 stderr = sqrt(diag(covmat)) --> standard errors for beta
 deviance = -2l              --> scaled deviance statistic
 chi-squared value for -2l is the model chi-squared test.
    """
    if x.shape[-1] != len(y):
        raise ValueError, "x.shape[-1] and y should be the same length!"
    try:
        N, npreds = x.shape[1], x.shape[0]
    except: # single predictor, use simple logistic regression routine.
        return _simple_logistic_regression(x,y,beta_start=beta_start,
               CONV_THRESH=CONV_THRESH,MAXIT=MAXIT,verbose=verbose)
    if beta_start is None:
        beta_start = NA.zeros(npreds+1,x.dtype.char)
    X = NA.ones((npreds+1,N), x.dtype.char)
    X[1:, :] = x
    Xt = NA.transpose(X)
    iter = 0; diff = 1.; beta = beta_start  # initial values
    if verbose:
        print 'iteration  beta log-likliehood |beta-beta_old|' 
    while iter < MAXIT:
        beta_old = beta 
        ebx = NA.exp(NA.dot(beta, X))
        p = ebx/(1.+ebx)
        l = NA.sum(y*NA.log(p) + (1.-y)*NA.log(1.-p)) # log-likeliehood
        s = NA.dot(X, y-p)                            # scoring function
        J_bar = NA.dot(X*p,Xt)                        # information matrix
        beta = beta_old + NA.dot(LA.inverse(J_bar),s) # new value of beta
        diff = NA.sum(NA.fabs(beta-beta_old)) # sum of absolute differences
        if verbose:
            print iter+1, beta, l, diff
        if diff <= CONV_THRESH: break
        iter = iter + 1
    if iter == MAXIT and diff > CONV_THRESH: 
        print 'warning: convergence not achieved with threshold of %s in %s iterations' % (CONV_THRESH,MAXIT)
    return beta, J_bar, l

def calcprob(beta, x):
    """
 calculate probabilities (in percent) given beta and x
    """
    try:
        N, npreds = x.shape[1], x.shape[0]
    except: # single predictor, x is a vector, len(beta)=2.
        N, npreds = len(x), 1
    if len(beta) != npreds+1:
        raise ValueError,'sizes of beta and x do not match!'
    if npreds==1: # simple logistic regression
        return 100.*NA.exp(beta[0]+beta[1]*x)/(1.+NA.exp(beta[0]+beta[1]*x))
    X = NA.ones((npreds+1,N), x.dtype.char)
    X[1:, :] = x
    ebx = NA.exp(NA.dot(beta, X))
    return 100.*ebx/(1.+ebx)

if __name__ == '__main__':
    # this example uses three correlated time series drawn from 
    # a trivariate normal distribution.  The first is taken to be the
    # observations, the other two are considered to be forecasts
    # of the observations.  For example, the observations could
    # be the temperature in Boulder, and the other two could
    # be forecasts of temperature from two different weather prediction
    # models. A logistic regression is used to compute
    # the conditional probability that the observation will be greater
    # than zero given the forecasts.
    if _use_numarray:
        from numarray.random_array import multivariate_normal
        import numarray.mlab as mlab
    else:
        from numpy.oldnumeric.random_array import multivariate_normal
        import numpy.oldnumeric.mlab as mlab
    # number of realizations.
    nsamps = 100000
    # correlations
    r12 = 0.5 # average correlation between the first predictor and the obs.
    r13 = 0.25 # avg correlation between the second predictor and the obs.
    r23 = 0.125 # avg correlation between predictors.
    # random draws from trivariate normal distribution
    x = multivariate_normal(NA.array([0,0,0]),NA.array([[1,r12,r13],[r12,1,r23],[r13,r23,1]]), nsamps)
    x2 = multivariate_normal(NA.array([0,0,0]),NA.array([[1,r12,r13],[r12,1,r23],[r13,r23,1]]), nsamps)
    print 'correlations (r12,r13,r23) = ',r12,r13,r23
    print 'number of realizations = ',nsamps
    # training data.
    obs = x[:,0]
    climprob = NA.sum((obs > 0).astype('f'))/nsamps
    fcst = NA.transpose(x[:,1:]) # 2 predictors.
    obs_binary = obs > 0.
    # independent data for verification.
    obs2 = x2[:,0]
    fcst2 = NA.transpose(x2[:,1:])
    # compute logistic regression.
    beta,Jbar,llik = logistic_regression(fcst,obs_binary,verbose=True)
    covmat = LA.inverse(Jbar)
    stderr = NA.sqrt(mlab.diag(covmat))
    print 'beta =' ,beta
    print 'standard error =',stderr
    # forecasts from independent data.
    prob = calcprob(beta, fcst2)
    # compute Brier Skill Score
    verif = (obs2 > 0.).astype('f')
    bs = mlab.mean((0.01*prob - verif)**2)
    bsclim = mlab.mean((climprob - verif)**2)
    bss = 1.-(bs/bsclim)
    print 'Brier Skill Score (should be within +/- 0.1 of 0.18) = ',bss
    # calculate reliability. 
    # see http://www.bom.gov.au/bmrc/wefor/staff/eee/verif/verif_web_page.html
    # for information on the Brier Skill Score and reliability diagrams.
    totfreq = NA.zeros(10,'f')
    obfreq = NA.zeros(10,'f')
    for icat in range(10):
        prob1 = icat*10.
        prob2 = (icat+1)*10.
        test1 = prob > prob1
        test2 = prob <= prob2
        testf = 1.0*test1*test2
        testfv = verif*testf 
        totfreq[icat] = NA.sum(testf)
        obfreq[icat] = NA.sum(testfv)
    fcstprob = NA.zeros(10,'f')
    reliability = NA.zeros(10,'f')
    frequse = NA.zeros(10,'f')
    print 'fcst prob, reliability, frequency of use'
    for icat in range(10):
        prob1 = icat*10.
        prob2 = (icat+1)*10.
        fcstprob[icat] = 0.5*(prob1+prob2)
        reliability[icat]=1.e20
        if totfreq[icat] > nsamps/1000.:
            reliability[icat] = 100.*obfreq[icat]/totfreq[icat]
        frequse[icat] = 100.*totfreq[icat]/nsamps
        print fcstprob[icat],reliability[icat],frequse[icat]\
    # plot reliability diagram if matplotlib installed.
    try:
        from pylab import *
        doplot = True
    except:
        doplot = False
    if doplot:
        from matplotlib.numerix import ma
        reliability = ma.masked_values(reliability, 1.e20)
        fig=figure(figsize=(6.5,6.5))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        plot(fcstprob,reliability,'bo-')
        plot(arange(0,110,10),arange(0,110,10),'r--')
        xlabel('forecast probability')
        ylabel('observed frequency')
        title('Reliability Diagram')
        text(55,15,'Brier Skill Score = %4.2f' % bss,fontsize=14)
        ax2 = fig.add_axes([.2, .6, .25, .2], axisbg='y')
        bar(10*arange(10), frequse, width=10)
        xlabel('forecast probability',fontsize=10)
        ylabel('percent issued',fontsize=10)
        title('Frequency of Use',fontsize=12)
        ax2.set_xticklabels(arange(20,120,20),fontsize=9)
        ax2.set_yticklabels(arange(20,120,20),fontsize=9)
        ax.set_xticks(arange(5,100,5)); ax.set_yticks(arange(5,100,5))
        ax.grid(True)
        print 'saving reliability diagram ...'
        savefig('reliability')
