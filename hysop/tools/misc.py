"""Some utilities to deal with slices and other python objects
"""

class utils(object):

    @staticmethod
    def arrayToDict(inarray):
        """
        convert an array into a dictionnary,
        keys being the column numbers in array
        and values the content of each corresponding column
        transformed into a list of slices like this:
        column = [1, 4, 2, 6, ...] ---> [slice(1, 4), slice(2, 6), ...]
        """
        outslice = {}
        size = inarray.shape[1]
        dimension = (int)(0.5 * inarray.shape[0])
        for rk in xrange(size):
            outslice[rk] = [slice(inarray[2 * d, rk],
                                  inarray[2 * d + 1, rk] + 1)
                            for d in xrange(dimension)]
        return outslice

    @staticmethod
    def intersl(sl1, sl2):
        """Intersection of two lists of slices

        Parameters
        -----------
        sl1 : a list of slices
        sl2 : a list of slices

        Return :
            guess what ... a list of slices such that:
            result[i] = intersection between sl1[i] and sl2[i]
        """
        assert len(sl1) == len(sl2)
        res = [None] * len(sl1)
        for d in xrange(len(sl1)):
            s1 = sl1[d]
            s2 = sl2[d]
            start = max(s1.start, s2.start)
            stop = min(s1.stop, s2.stop)
            if stop <= start:
                return None
            res[d] = slice(start, stop)
        return tuple(res)
