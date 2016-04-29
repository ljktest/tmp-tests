"""
@file method.py

Abstract interface to numerical methods.
"""
from abc import ABCMeta, abstractmethod
from hysop.constants import debug


class NumMethod(object):
    """ Abstract description of numerical method. """

    __metaclass__ = ABCMeta

    @debug
    def __new__(cls, *args, **kw):
        return object.__new__(cls, *args, **kw)

    @staticmethod
    def getWorkLengths(nb_components=None, domain_dim=None):
        """
        Compute the number of required work arrays for this method.
        @param nb_components : number of components of the
        @param domain_dim : dimension of the domain
        fields on which this method operates.
        @return length of list of work arrays of reals.
        @return length of list of work arrays of int.
        """
        return 0, 0

    @debug
    @abstractmethod
    def __call__(self):
        """Call the method"""
