from hysop.numerics.update_ghosts import UpdateGhosts, UpdateGhostsFull
from hysop.domain.box import Box
from hysop.fields.continuous import Field
from hysop.tools.parameters import Discretization
import numpy as np


def _setup(ghosts, topo_dim):
    dom = Box()

    f = Field(dom, is_vector=True, name='f')
    resolTopo = Discretization([33, 33, 33], ghosts=ghosts)
    topo = dom.create_topology(resolTopo, dim=topo_dim)
    f.discretize(topo)
    df = f.discreteFields[topo]
    for i, dfd in enumerate(df.data):
        dfd[...] = 0.
        dfd[topo.mesh.iCompute] = 1. * (i + 1)
    return df, topo


def verify(df, gh):
    sli = (
        (slice(0, gh[0]), slice(gh[1], -gh[1]), slice(gh[2], -gh[2])),
        (slice(-gh[0], None), slice(gh[1], -gh[1]), slice(gh[2], -gh[2])),
        (slice(gh[0], -gh[0]), slice(0, gh[1]), slice(gh[2], -gh[2])),
        (slice(gh[0], -gh[0]), slice(-gh[1], None), slice(gh[2], -gh[2])),
        (slice(gh[0], -gh[0]), slice(gh[1], -gh[1]), slice(0, gh[2])),
        (slice(gh[0], -gh[0]), slice(gh[1], -gh[1]), slice(-gh[2], None)))
    slni = (
        (slice(0, gh[0]), slice(0, gh[1]), slice(None)),
        (slice(0, gh[0]), slice(None), slice(0, gh[2])),
        (slice(None), slice(0, gh[1]), slice(0, gh[2])),
        (slice(-gh[0], None), slice(-gh[1], None), slice(None)),
        (slice(-gh[0], None), slice(None), slice(-gh[2], None)),
        (slice(None), slice(-gh[1], None), slice(-gh[2], None)))
    for i, dfd in enumerate(df.data):
        for s in sli:
            assert np.max(dfd[s]) == np.min(dfd[s]) and \
                np.max(dfd[s]) == 1. * (i + 1)
        for s in slni:
            assert np.max(dfd[s]) == np.min(dfd[s]) and \
                np.max(dfd[s]) == 0.


def verify_full(df, gh):
    for i, dfd in enumerate(df.data):
        for s in ((slice(0, gh[0]), slice(None), slice(None)),
                  (slice(-gh[0], None), slice(None), slice(None)),
                  (slice(None), slice(0, gh[1]), slice(None)),
                  (slice(None), slice(-gh[1], None), slice(None)),
                  (slice(None), slice(None), slice(0, gh[2])),
                  (slice(None), slice(None), slice(-gh[2], None))):
            assert np.max(dfd[s]) == np.min(dfd[s]) and \
                np.max(dfd[s]) == 1. * (i + 1)
        assert np.max(dfd) == np.min(dfd) and \
            np.max(dfd) == 1. * (i + 1)


def test_update_ghosts_simple_1D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 1)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_simple_2D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 2)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_simple_3D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 3)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_1D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 1)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_2D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 2)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_3D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 3)
    update = UpdateGhosts(topo, 3)
    update(df.data)
    verify(df, gh)


def test_update_ghosts_full_simple_1D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 1)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)


def test_update_ghosts_full_simple_2D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 2)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)


def test_update_ghosts_full_simple_3D():
    gh = [1, 1, 1]
    df, topo, = _setup(gh, 3)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)


def test_update_ghosts_full_1D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 1)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)


def test_update_ghosts_full_2D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 2)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)


def test_update_ghosts_full_3D():
    gh = [2, 3, 4]
    df, topo, = _setup(gh, 3)
    update = UpdateGhostsFull(topo, 3)
    update(df.data)
    verify_full(df, gh)

