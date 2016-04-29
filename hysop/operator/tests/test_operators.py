# -*- coding: utf-8 -*-
"""
@file test_operators.py
tests for operators general interface
"""

from hysop.mpi.tests.utils import create_multitask_context, CPU, GPU, OTHER
from hysop.tools.parameters import Discretization
from hysop.operator.analytic import Analytic
from hysop.mpi import main_size
import hysop as pp

r_ng = Discretization([33, ] * 3)


def v3d(res, x, y, z, t):
    res[0][...] = x + t
    res[1][...] = y
    res[2][...] = z
    return res


def test_tasks():
    """
    test proper tasks assignment
    """
    dom, topo = create_multitask_context(dim=3, discr=r_ng)
    assert topo.task_id() == dom.current_task()
    velo = pp.Field(domain=dom, name='velocity',
                    is_vector=True, formula=v3d)
    op = Analytic(variables={velo: r_ng})
    op.discretize()
    op.setup()

    assert op.task_id() == dom.current_task()
    if main_size == 8:
        if dom.is_on_task(CPU):
            assert op.variables[velo].size == 5
        elif dom.is_on_task(GPU):
            assert op.variables[velo].size == 2
        elif dom.is_on_task(OTHER):
            assert op.variables[velo].size == 1
