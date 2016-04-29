# -*- coding: utf-8 -*-
"""Tests for hysop io
"""

from hysop.tools.io_utils import IO, IOParams
import os


# IO params
def test_io_params_1():
    filename = 'toto.h5'
    dirname = './test/'
    iop = IOParams(filename, dirname)

    realdirname = os.path.abspath('./test/toto.h5')
    assert os.path.abspath(iop.filename) == realdirname
    assert iop.filepath == os.path.dirname(realdirname)


def test_io_params_2():
    filename = 'toto.h5'
    iop = IOParams(filename)
    def_path = IO.default_path()
    realdirname = os.path.join(def_path, filename)
    assert iop.filename == realdirname
    assert iop.filepath == os.path.dirname(realdirname)


def test_io_params_3():
    filename = './test/toto.h5'
    iop = IOParams(filename)
    realdirname = os.path.abspath('./test/toto.h5')
    assert iop.filename == realdirname
    assert iop.filepath == os.path.dirname(realdirname)


def test_io_params_4():
    filename = './test/toto.h5'
    dirname = './test2/'
    iop = IOParams(filename, dirname)

    realdirname = os.path.abspath('./test2/./test/toto.h5')
    assert os.path.abspath(iop.filename) == realdirname
    assert iop.filepath == os.path.dirname(realdirname)


def test_io_params_5():
    filename = '/tmp/test/toto.h5'
    iop = IOParams(filename)
    assert iop.filename == filename
    assert iop.filepath == os.path.dirname(filename)


def test_io_params_6():
    filename = '/tmp/toto.h5'
    dirname = '/test2/tmp'
    iop = IOParams(filename, dirname)
    assert iop.filename == filename
    assert iop.filepath == os.path.dirname(filename)

