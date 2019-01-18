import xarray as xr
import numpy as np
import operator as op


def tup2str(tup):
    return ','.join(map(str, tup))


def str2tup(s):
    return tuple(','.split(s))


def empty_array(coords):
    dims = list(sorted(coords.keys()))
    data = np.empty([len(v) for k, v in sorted(coords.items(), key=op.itemgetter(0))])
    data[:] = np.nan
    return xr.DataArray(data=data, dims=dims, coords=coords)


def _list2str(l):
    return str(l) if len(l) != 1 else str(l[0])


def make_title(dag_config, sample_config, alg_config, nnodes=True, nneighbors=True, nsamples=True, ntargets=True, nsettings=True):
    pieces = []
    if nnodes:
        pieces.append('nnodes=%s' % str(dag_config.nnodes))
    if nneighbors:
        pieces.append('nneighbors=%s' % _list2str(dag_config.nneighbors_list))
    if nsamples:
        pieces.append('nsamples=%s' % _list2str(sample_config.nsamples_list))
    if ntargets:
        pieces.append('ntargets=%s' % _list2str(sample_config.ntargets_list))
    if nsettings:
        pieces.append('nsettings=%s' % _list2str(sample_config.nsettings_list))
    return ';'.join(pieces)


def shd_mat(mat1, mat2):
    mat1 = mat1.copy()
    mat2 = mat2.copy()
    skel1 = mat1 + mat1.T
    skel2 = mat2 + mat2.T
    skel1[skel1 == 2] = 1
    skel2[skel2 == 2] = 1
    skel_diff = skel1 - skel2

    ixs_skel1 = np.where(skel_diff > 0)
    mat1[ixs_skel1] = 0

    ixs_skel2 = np.where(skel_diff < 0)
    mat1[ixs_skel2] = mat2[ixs_skel2]

    d = np.abs(mat1 - mat2)
    return len(ixs_skel1[0])/2 + len(ixs_skel2[0])/2 + np.sum(d + d.T > 0)/2


if __name__ == '__main__':
    d0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    d1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    u1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    d2 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    u2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    d3 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])

    print(shd_mat(d1, u1), 1)
    print(shd_mat(d1, d2), 1)
    print(shd_mat(d2, u2), 2)
    print(shd_mat(d0, d1), 1)
    print(shd_mat(d0, u1), 1)
    print(shd_mat(d0, d2), 2)
    print(shd_mat(d0, u2), 2)
    print(shd_mat(d0, d3), 3)
