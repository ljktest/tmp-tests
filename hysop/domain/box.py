"""Box-shaped domains definition.
"""
from hysop.domain.domain import Domain
from hysop.constants import PERIODIC, debug
import hysop.tools.numpywrappers as npw


class Box(Domain):
    """ Box-shaped domain description.

    todo implement different boundary conditions typesKHave different BC
    """

    @debug
    def __init__(self, length=None, origin=None, **kwds):
        """
        Create a Periodic Box from a dimension, length and origin.

        Parameters
        -----------
        length : list or numpy array of double
            box sides lengthes. Default = [1.0, ...]
        origin : list or numpy array of double
            position of the lowest point of the box. Default [0., ...]

        Example:

        >>> import hysop as pp
        >>> import numpy as np
        >>> b = pp.Box()
        >>> (b.end == np.asarray([1.0, 1.0, 1.0])).all()
        True

        """
        if 'dimension' not in kwds:
            if length is not None or origin is not None:
                dim = [len(list(j)) for j in [length, origin]
                       if j is not None]
                kwds['dimension'] = dim[0]

        super(Box, self).__init__(**kwds)

        ##  Box length.
        if length is None:
            length = [1.0] * self.dimension
        if origin is None:
            origin = [0.] * self.dimension
        self.length = npw.const_realarray(length)
        ##  Box origin
        self.origin = npw.const_realarray(origin)

        ## Maximum Box position. max = origin + length
        self.end = self.origin + self.length
        # set periodic boundary conditions
        if self.boundaries is None:
            self.boundaries = npw.asdimarray([PERIODIC] * self.dimension)
        else:
            msg = 'Boundary conditions must be periodic.'
            assert list(self.boundaries).count(PERIODIC) == self.dimension, msg

    def __str__(self):
        s = str(self.dimension) + \
            "D box (parallelepipedic or rectangular) domain : \n"

        s += "   origin : " + str(self.origin) + ", maxPosition :" \
             + str(self.end) + ", lengths :" + str(self.length) + "."
        return s

    def __eq__(self, other):
        c1 = (self.length == other.length).all()
        c2 = (self.origin == other.origin).all()
        c3 = (self.boundaries == other.boundaries).all()
        c4 = self.current_task() == other.current_task()
        return c1 and c2 and c3 and c4
