"""Light classes to handle parameters for classes construction.

.. currentmodule hysop.tools

* :class:`~MPIParams`
* :class:`~Discretization`

"""

from collections import namedtuple
from hysop.mpi.main_var import main_comm, main_rank, MPI
from hysop.constants import DEFAULT_TASK_ID
import hysop.tools.numpywrappers as npw


class MPIParams(namedtuple('MPIParams', ['comm', 'task_id',
                                         'rank', 'on_task'])):
    """
    Struct to save mpi parameters :
    - comm : parent mpi communicator (default = main_comm)
    - task_id : id of the task that owns this object
    (default = DEFAULT_TASK_ID)
    - rank of the current process in comm
    - on_task : true if the task_id of the object corresponds
    to the task_id of the current process.

    This struct is useful for operators : each operator has
    a MPIParams attribute to save its mpi settings.

    Examples
    ---------

    op = SomeOperator(..., task_id=1)
    if op.is_on_task():
       ...

    'is_on_task' will return MPIParams.on_task value for op
    and tell if the current operator belongs to the current process
    mpi task.
    """
    def __new__(cls, comm=main_comm, task_id=DEFAULT_TASK_ID,
                rank=main_rank, on_task=True):
        if comm != MPI.COMM_NULL:
            rank = comm.Get_rank()
        else:
            rank = MPI.UNDEFINED
        return super(MPIParams, cls).__new__(cls, comm, task_id,
                                             rank, on_task)


class Discretization(namedtuple("Discretization", ['resolution', 'ghosts'])):
    """
    A struct to handle discretization parameters:
    - a resolution (either a list of int or a numpy array of int)
    - number of points in the ghost-layer. One value per direction, list
    or array. Default = None.
    """
    def __new__(cls, resolution, ghosts=None):
        resolution = npw.asdimarray(resolution)
        if ghosts is not None:
            ghosts = npw.asintarray(ghosts)
            msg = 'Dimensions of resolution and ghosts parameters'
            msg += ' are not complient.'
            assert ghosts.size == resolution.size, msg
            assert all(ghosts >= 0)
        else:
            ghosts = npw.int_zeros(resolution.size)
        return super(Discretization, cls).__new__(cls, resolution, ghosts)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        return (self.resolution == other.resolution).all() and\
            (self.ghosts == other.ghosts).all()

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result
