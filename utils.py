import xarray as xr
import numpy as np
import operator as op


def tup2str(tup):
    return ','.join(map(str, tup))


def str2tup(s):
    return tuple(','.split(s))


def empty_array(coords):
    dims = list(sorted(coords.keys()))
    data = np.zeros([len(v) for k, v in sorted(coords.items(), key=op.itemgetter(0))])
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

