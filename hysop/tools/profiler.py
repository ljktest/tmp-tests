"""Tools to collect time profiling information
for hysop classes.
"""
from hysop.mpi import Wtime as ftime
from hysop.mpi.main_var import main_rank
import hysop.tools.numpywrappers as npw
import numpy as np


def profile(f):
    """
    Decorator to enable function profiling of
    a method inside a class. The concerned class
    MUST have a Profiler attribute.
    """
    def deco(*args, **kwargs):
        """args[0] contains the object"""
        t0 = ftime()
        res = f(*args, **kwargs)
        # add current method and its elapsed time
        # to the 'profiled' collection
        args[0].profiler[f.func_name] += ftime() - t0
        return res
    return deco


class FProfiler(object):
    """Class for time measurments on the fly.
    The objects can be linked to a class method.
    """
    def __init__(self, fname):
        """Object to profile on the fly sections of code, methods ...

        Usage:

        >>> from hysop.tools.profiler FProfiler, ftime
        >>> prof = FProfiler('some_name')
        >>> start = ftime()
        >>> # do something ... [1]
        >>> prof += ftime() - start
        >>> # do something else ... [2]
        >>> prof += ftime() - start
        >>> # ...
        >>> print prof
        >>> # --> display total time spent in do [1] and [2]
        >>> # and number of calls of prof

        """
        # Function name
        self.fname = fname
        # Total execution time
        self.total_time = 0.
        # Number of calls of the fprofiler
        self.nb_calls = 0

    def get_name(self):
        """Profiler name"""
        return self.fname

    def __iadd__(self, t):
        """+= operator"""
        self.total_time += t
        self.nb_calls += 1
        return self

    def __str__(self):
        if self.nb_calls > 0:
            s = "- {0} : {1} s ({2} call{3})".format(
                self.fname, self.total_time, self.nb_calls,
                's' if self.nb_calls > 1 else '')
        else:
            s = ""
        return s


class Profiler(object):
    """Object used to collect profiling information inside operators.
    """

    def __init__(self, obj, comm):
        """Collect profiling information for all operator
        method decorate with @profile

        Parameters
        ----------
        obj : object (python class) instance. See requirements in notes below.

        Notes:

        * obj must have 'get_profiling' and 'name'
        attribute/method.
        """
        self.summary = {}
        self.table = []
        self._comm = comm
        self._comm_size = comm.Get_size()
        for _ in xrange(self._comm_size):
            self.table.append([])
        # A dictionnary of profiled functions/methods as keys
        # and elapsed time as value.
        self._elems = {}
        # profiled object
        self._obj = obj
        # ?
        self._l = 1
        self.all_times = None
        self.all_call_nb = None
        self.all_names = [None]


    def down(self, l):
        self._l = l + 1

    def get_name(self):
        """Return the name of the profiled object
        """
        try:
            _name = self._obj.name
        except AttributeError:
            if isinstance(self._obj, str):
                _name = self._obj
            else:
                _name = 'unknown'
        return _name

    def __iadd__(self, other):
        """+= operator. Append a new profiled function to the collection"""
        self._elems[other.get_name()] = other
        return self

    def __setitem__(self, key, value):
        self._elems[key] = value

    def __getitem__(self, item):
        try:
            return self._elems[item]
        except KeyError:
            self._elems[item] = FProfiler(item)
            return self._elems[item]

    def __str__(self):
        if len(self.summary) > 0:
            s = ""
            if self._l == 1:
                s += "\n[{0}]".format(main_rank)
            s += ' - ' + self.get_name() + ":"
            for k in sorted(self.summary):
                if len(str(self.summary[k])) > 0:
                    s += "\n[{0}]".format(main_rank)
                    s += '  ' * self._l + str(self.summary[k])
        else:
            s = ""
        return s

    def write(self, prefix='', hprefix='', with_head=True):
        """

        Parameters
        ----------
        prefix : string, optional
        hprefix : string, optional
        with_head : bool, optional

        """
        if prefix != '' and prefix[-1] != ' ':
            prefix += ' '
        if hprefix != '' and hprefix[-1] != ' ':
            hprefix += ' '
        if self._comm.Get_rank() == 0:
            s = ""
            h = hprefix + "Rank"
            for r in xrange(self._comm_size):
                s += prefix + "{0}".format(r)
                for i in xrange(len(self.all_names)):
                    s += " {0}".format(self.all_times[i][r])
                s += "\n"
            s += prefix + "-1"
            for i in xrange(len(self.all_names)):
                h += ' ' + self.all_names[i]
                s += " {0}".format(self.all_times[i][self._comm_size])
            h += "\n"
            if with_head:
                s = h + s
            print s

    def summarize(self):
        """Update profiling values and prepare data for a report
        with print or write.
        """
        # reset summary
        self.summary = {}

        # collect profiling results from decorated object(s), if any.
        try:
            self._obj.get_profiling_info()
        except AttributeError:
            pass

        from hysop.fields.continuous import Field
        i = 0
        for k in self._elems.keys():
            try:
                # Either elem[k] is a FProfiler ...
                self.summary[self._elems[k].total_time] = self._elems[k]
            except AttributeError:
                # ... or a Profiler
                i += 1
                self._elems[k].down(self._l)
                self._elems[k].summarize()
                if isinstance(self._elems[k]._obj, Field):
                    self.summary[1e10 * i] = self._elems[k]
                else:
                    self.summary[1e8 * i] = self._elems[k]
        rk = self._comm.Get_rank()
        for k in sorted(self._elems.keys()):
            if isinstance(self._elems[k], FProfiler):
                self.table[rk].append(
                    (self.get_name() + '.' + k,
                     self._elems[k].total_time, self._elems[k].nb_calls))
        for k in sorted(self._elems.keys()):
            if isinstance(self._elems[k], Profiler):
                for e in self._elems[k].table[rk]:
                    self.table[rk].append(
                        (self.get_name() + '.' + e[0], e[1], e[2]))

        if self._l == 1 and self.all_times is None:
            nb = len(self.table[rk])
            self.all_times = npw.zeros((nb, self._comm_size + 1))
            self.all_call_nb = npw.int_zeros((nb, self._comm_size + 1))
            self.all_names = [None] * nb
            for i in xrange(nb):
                self.all_names[i] = self.table[rk][i][0]
                self.all_times[i, rk] = self.table[rk][i][1]
                self.all_call_nb[i, rk] = self.table[rk][i][2]
            for p in xrange(self._comm_size):
                self._comm.Bcast(self.all_times[:, p], root=p)
                self._comm.Bcast(self.all_call_nb[:, p], root=p)
            self.all_times[:, self._comm_size] = np.sum(
                self.all_times[:, :self._comm_size], axis=1)
            self.all_call_nb[:, self._comm_size] = np.sum(
                self.all_call_nb[:, :self._comm_size], axis=1)
