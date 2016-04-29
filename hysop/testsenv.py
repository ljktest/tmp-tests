"""Set some functions and variables useful to run tests.
"""
from hysop import __FFTW_ENABLED__, __GPU_ENABLED__, __SCALES_ENABLED__
import pytest
import shutil
from hysop.tools.io_utils import IO
from hysop.mpi import main_rank
import os

# accept failing tests when fft is not enabled
if __FFTW_ENABLED__:
    def fftw_failed(f):
        """For fftw tests that must not fail
        """
        return f

else:
    fftw_failed = pytest.mark.xfail


# accept failing tests when scales is not enabled
if __SCALES_ENABLED__:
    def scales_failed(f):
        """For scales tests that must not fail
        """
        return f

else:
    scales_failed = pytest.mark.xfail


hysop_failed = pytest.mark.xfail
"""Use this decorator for tests that must fail"""


class postclean(object):
    """A decorator to remove files in default path and working dir
       at the end of the calling function.
    """

    def __init__(self, working_dir=None):
        """A decorator to remove files in default path and working dir
           at the end of the calling function.

           Usage:

           @postclean
           def test_name()

           or

           @postclean(working_dir)
           def test_name()

           working_dir = current test working directory.
        """
        if working_dir is not None:
            if not os.path.exists(working_dir):
                working_dir = None
        self.working_dir = working_dir

    def __call__(self, f):
        """Apply decorator
        """
        def wrapped_f(*args):
            """return wrapped function + post exec
            """
            f(*args)
            print "RM ...", self.working_dir, IO.default_path(), main_rank
            if main_rank == 0:
                if os.path.exists(IO.default_path()):
                    shutil.rmtree(IO.default_path())
                if self.working_dir is not None:
                    if os.path.exists(self.working_dir):
                        shutil.rmtree(self.working_dir)
        return wrapped_f
