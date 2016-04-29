"""Tools related to global config.
"""
import sys


class SysUtils(object):
    """
    Global system check and other utilities
    """
    @staticmethod
    def in_ipython():
        """True if current session is run under ipython
        """
        try:
            __IPYTHON__
        except NameError:
            return False
        else:
            return True

    @staticmethod
    def is_interactive():
        """True if run under interactive session, else
        False. Warning : return true in any case if ipython is
        used.
        """
        return hasattr(sys, 'ps1') or hasattr(sys, 'ipcompleter')
