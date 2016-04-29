from hysop.domain.box import Box
from hysop.tools.parameters import Discretization
from hysop.mpi.bridge import Bridge
from hysop.mpi.bridge_overlap import BridgeOverlap
from hysop.mpi.bridge_inter import BridgeInter
import math


def test_bridge2D():
    dimension = 2
    Lx = Ly = 2 * math.pi
    dom = Box(length=[Lx, Ly], origin=[0., 0.])
    # Global discretization, no ghosts
    r_ng = Discretization([33, ] * dimension)
    # Global discretization, with ghosts
    r_wg = Discretization([33, ] * dimension, [0, 1])
    source = dom.create_topology(discretization=r_ng,
                                 cutdir=[False, True])
    target = dom.create_topology(discretization=r_wg)
    bridge = Bridge(source, target)
    # We cannot really check something interesting,
    # so we just create a bridge.
    # The real tests are done in test_redistribute.py
    print bridge


def test_bridge3D():
    dimension = 3
    Lx = 2 * math.pi
    dom = Box(length=[Lx, ] * dimension, origin=[0., ] * dimension)
    # Global discretization, no ghosts
    r_ng = Discretization([33, ] * dimension)
    # Global discretization, with ghosts
    r_wg = Discretization([33, ] * dimension, [0, 1, 2])
    source = dom.create_topology(discretization=r_ng,
                                 cutdir=[False, False, True])
    target = dom.create_topology(discretization=r_wg)
    bridge = Bridge(source, target)
    # We cannot really check something interesting,
    # so we just create a bridge.
    # The real tests are done in test_redistribute.py
    print bridge


from hysop.mpi.main_var import main_size, main_comm
from hysop.mpi.tests.utils import create_subtopos, create_inter_topos


def test_bridge_overlap():
    """
    Try the pathologic case where source and target do not apply on
    the same group of process but when groups overlap.
    """

    if main_size < 4:
        return
    dimension = 3
    # Global discretization, no ghosts
    r_ng = Discretization([33, ] * dimension)
    # Global discretization, with ghosts
    r_wg = Discretization([33, ] * dimension, [0, 1, 2])
    Lx = 2 * math.pi
    dom = Box(length=[Lx, ] * dimension, origin=[0., ] * dimension)
    source_topo, target_topo = create_subtopos(dom, r_ng, r_wg)
    bridge = BridgeOverlap(source=source_topo, target=target_topo,
                           comm_ref=main_comm)
    assert bridge is not None


def test_bridgeInter2D():
    if main_size < 4:
        return

    dimension = 2
    # Global discretization, no ghosts
    r_ng = Discretization([33, ] * dimension)
    # Global discretization, with ghosts
    r_wg = Discretization([33, ] * dimension, [0, 1])
    dom, topo1, topo2 = create_inter_topos(2, r_ng, r_wg)
    CPU = 1
    GPU = 4
    bridge = BridgeInter(topo1, main_comm, source_id=CPU, target_id=GPU)
    tr = bridge.transferTypes()
    assert bridge is not None
    assert isinstance(tr, dict)
    # We cannot really check something interesting,
    # so we just create a bridge.
    # The real tests are done in test_redistribute.py

    # Bridge from topo2 on GPU to topo1 on CPU:
    if dom.is_on_task(GPU):
        bridge2 = BridgeInter(topo2, main_comm, source_id=GPU, target_id=CPU)
    elif dom.is_on_task(CPU):
        bridge2 = BridgeInter(topo1, main_comm, source_id=GPU, target_id=CPU)
    assert bridge2 is not None


def test_bridgeInter3D():
    if main_size < 4:
        return
    dimension = 3
    # Global discretization, no ghosts
    r_ng = Discretization([33, ] * dimension)
    # Global discretization, with ghosts
    r_wg = Discretization([33, ] * dimension, [0, 1, 2])
    dom, topo1, topo2 = create_inter_topos(3, r_ng, r_wg)
    CPU = 1
    GPU = 4
    bridge = BridgeInter(topo1, main_comm, source_id=CPU, target_id=GPU)
    tr = bridge.transferTypes()
    assert bridge is not None
    assert isinstance(tr, dict)
    # We cannot really check something interesting,
    # so we just create a bridge.
    # The real tests are done in test_redistribute.py
    # Bridge from topo2 on GPU to topo1 on CPU:
    if dom.is_on_task(GPU):
        bridge2 = BridgeInter(topo2, main_comm, source_id=GPU, target_id=CPU)
    elif dom.is_on_task(CPU):
        bridge2 = BridgeInter(topo1, main_comm, source_id=GPU, target_id=CPU)
    assert bridge2 is not None

