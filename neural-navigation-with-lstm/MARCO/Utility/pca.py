## Automatically adapted for numpy.oldnumeric May 17, 2011 by -c

# Warning: hackish and not properly tested ripped out bit of code ahead
# so no guarantees whatsoever
# Anyway, it should at least sort of give you the idea
# try pca(X); if that doesn't do what you want try pca(t(X))

from numpy.oldnumeric import take, dot, shape, argsort, where, sqrt, transpose as t
from numpy.oldnumeric.linear_algebra import eigenvectors

def pca(M):
    "Perform PCA on M, return eigenvectors and eigenvalues, sorted."
    T, N = shape(M)
    # if there are fewer rows T than columns N, use snapshot method
    if T < N:
        C = dot(M, t(M))
        evals, evecsC = eigenvectors(C)
        # HACK: make sure evals are all positive
        evals = where(evals < 0, 0, evals)
        evecs = 1./sqrt(evals) * dot(t(M), t(evecsC))
    else:
        # calculate covariance matrix
        K = 1./T * dot(t(M), M)
        evals, evecs = eigenvectors(K)
    # sort the eigenvalues and eigenvectors, descending order
    order = (argsort(evals)[::-1])
    evecs = take(evecs, order, 1)
    evals = take(evals, order)
    return evals, t(evecs)

