import numpy as np


def condition2Slice(cond):
    dim = len(cond.shape)
    ilist = np.where(cond)
    if ilist[0].size == 0:
        isEmpty = True
        sl = [slice(0, 0) for i in xrange(dim)]
        resol = np.asarray([0] * dim)
    else:
        start = np.asarray([ilist[i].min() for i in xrange(dim)])
        end = np.asarray([ilist[i].max() + 1 for i in xrange(dim)])
        sl = [slice(start[i], end[i])
              for i in xrange(dim)]
        resol = end - start
        isEmpty = False

    return sl, isEmpty, resol


def removeLastPoint(cond):

    shape = cond.shape
    dim = len(shape)
    ilist = np.where(cond)
    end = [ilist[i].max() for i in xrange(dim)]
    subl = [np.where(ilist[i] == end[i]) for i in xrange(dim)]
    for sl in subl:
        sublist = [ilist[i][sl] for i in xrange(dim)]
        sublist = tuple(sublist)
        cond[sublist] = False
    return cond
