"""
@file QtRendering.py

Contains all stuff to perform real-time rendering on GPU.
"""
from hysop.constants import debug, np, HYSOP_REAL
import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import QGLWidget
import OpenGL.GL as gl
from hysop.gpu.tools import get_opengl_shared_environment
from hysop.gpu import cl
from hysop.gpu.gpu_discrete import GPUDiscreteField
from hysop.gpu.gpu_kernel import KernelLauncher
from hysop.mpi import main_rank
from hysop.operator.computational import Computational
import hysop.tools.numpywrappers as npw


class QtOpenGLRendering(Computational):
    """
    Monitor that performs the rendering.

    Rendering is handled by OpenGL instructions. Context is shared between
    OpenGL and OpenCL.

    Vertex Buffer Objects are created on the OpenGL side and bound to OpenCL
    GLBuffers. VBOs store points coordinates and color that contains
    respectively X,Y coordinates and RGBA color definition. An OpenCL kernel
    computes once the coordinates and an other colorize regarding values of
    a given scalar field.

    Redering is displayed asynchronously. Main tread handle Qt application
    execution. A secondary thread, a QThread, performs the computations loop.
    A redrawing singal is emmited to synchronize threads.

    @remark Rendering is implemented only for 2D problems
    @remark Rendering is implemented only in single precision

    @see http://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
    @see http://enja.org/2011/03/22/adventures-in-pyopencl-part-2-particles-with-pyopengl/
    """

    @debug
    def __init__(self, field, component=0):
        """
        Build an OpenGL rendering object.

        @param field : Scalar field to render.

        Store a QApplication and a QMainWindow objects.
        """
        super(QtOpenGLRendering, self)._init__([field],
                                               frequency=1, name="QtRendering")
        if not field.dimension == 2:
            raise ValueError("Rendering implemented in 2D only.")
        ## Qt application
        self.app = QtGui.QApplication(sys.argv)
        ## Visualization window
        self.window = TestWindow()
        self.isGLRender = True
        self.input = [field]
        self.component = component if field.nb_components > 1 else 0
        self.output = []
        self.ctime = 0.
        self.mtime = 0.

    @debug
    def setup(self):
        """
        Create two VBOs buffers: GL_STATIC_DRAW and GL_COLOR_ARRAY.
        Create two OpenCL GLBuffers bound to VBOs.
        Pass buffers to QGLWidget and compile OpenCL kernels.
        """
        ## GPU scalar field
        for df in self.variables[0].discreteFields.values():
            if isinstance(df, GPUDiscreteField):
                self.gpu_field = df
        # Create OpenGL VBOs
        ## VBO for coordinates
        if self.gpu_field.nb_components > 1:
            self.pos_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pos_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.gpu_field.data[self.component].nbytes * 2,
                            None, gl.GL_STATIC_DRAW)  # gl.GL_DYNAMIC_DRAW
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)
            ## VBO for color
            self.color_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.gpu_field.data[self.component].nbytes * 4,
                            None, gl.GL_STREAM_DRAW)  # gl.GL_DYNAMIC_DRAW
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(4, gl.GL_FLOAT, 0, None)
        else:
            self.pos_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pos_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.gpu_field.data[0].nbytes * 2,
                            None, gl.GL_STATIC_DRAW)  # gl.GL_DYNAMIC_DRAW
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)
            ## VBO for color
            self.color_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.gpu_field.data[0].nbytes * 4,
                            None, gl.GL_STREAM_DRAW)  # gl.GL_DYNAMIC_DRAW
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(4, gl.GL_FLOAT, 0, None)

        # Create OpenCL GLBuffers
        ## OpenCL GLBuffer for coordinates
        self.pos = cl.GLBuffer(
            self.window.widget.cl_env.ctx, cl.mem_flags.READ_WRITE,
            int(self.pos_vbo))
        ## OpenCL GLBuffer for color
        self.color = cl.GLBuffer(
            self.window.widget.cl_env.ctx, cl.mem_flags.READ_WRITE,
            int(self.color_vbo))
        # Pass VBO and GLBuffers to the QGLWidget
        self.window.widget.setup(gl_buffers=[self.pos, self.color],
                                 color_vbo=self.color_vbo,
                                 pos_vbo=self.pos_vbo,
                                 partNumber=np.prod(
                self.gpu_field.topology.mesh.resolution))

        total_mem_used = self.pos.size + \
            self.color.size
        print "Total Device Global Memory used  for rendering: ",
        print total_mem_used, "Bytes (", total_mem_used / (1024 ** 2), "MB)",
        print "({0:.3f} %)".format(
            100 * total_mem_used / (
                self.window.widget.cl_env.device.global_mem_size * 1.))
        ## OpenCL kernel binaries
        self.prg = self.window.widget.cl_env.build_src(
            'kernels/rendering.cl')
        ## OpenCL kernel for computing coordinates
        if self.gpu_field.nb_components > 1:
            gwi = self.gpu_field.data[self.component].shape
        else:
            gwi = self.gpu_field.data[0].shape
        self.initCoordinates = KernelLauncher(
            self.prg.initPointCoordinates, self.window.widget.cl_env.queue,
            gwi, None)
        ## OpenCL kernel for computing colors
        self.numMethod = KernelLauncher(
            self.prg.colorize, self.window.widget.cl_env.queue,
            gwi, None)

        self.window.show()
        ## Text label of the window StatusBar
        self.labelText = str(self.gpu_field.topology.mesh.resolution)
        self.labelText += " particles, "
        coord_min = npw.ones(4)
        mesh_size = npw.ones(4)
        coord_min[0:2] = npw.asrealarray(self.gpu_field.topology.mesh.origin)
        mesh_size[0:2] = npw.asrealarray(self.gpu_field.topology.mesh.space_step)
        self.initCoordinates(self.pos, coord_min, mesh_size)

    @debug
    def apply(self, simulation):
        """
        Update the color GLBuffer and redraw the QGLWidget.
        """
        t = simulation.time
        dt = simulation.timeStep
        if main_rank == 0:
            simulation.printState()
        # OpenCL update
        self.numMethod(self.gpu_field.gpu_data[self.component],
                       self.color)
        self.window.widget.updateGL()
        if simulation.currentIteration > 1:
            self.window.label.setText(
                self.labelText + "t={0:6.2f}, fps={1:6.2f}".format(
                    t + dt,
                    1. / (self.timer.f_timers.values()[0].t - self.ctime)))
        self.ctime = self.timer.f_timers.values()[0].t

    @debug
    def finalize(self):
        """
        Terminates the thread containing the computations loop.
        """
        self.thread.quit()
        self.color.release()
        self.pos.release()

        if self.initCoordinates.f_timer is not None:
            for f_timer in self.initCoordinates.f_timer:
                self.timer.addFunctionTimer(f_timer)
        if self.numMethod.f_timer is not None:
            for f_timer in self.numMethod.f_timer:
                self.timer.addFunctionTimer(f_timer)

    @debug
    def startMainLoop(self):
        """
        Starts the secondary thread, that handle computation loop, and the main
        Qt application.
        """
        self.thread.start()
        self.theMainLoop.emit(QtCore.SIGNAL("step()"))
        self.app.exec_()

    def setMainLoop(self, problem):
        """
        Set a secondary QThread to performs problem solve computations.
        Synchronims between threads is done by signal emmission.
        @param problem : Problem to set the Qt main loop.
        """
        def problem_step():
            if not problem.simulation.isOver:
                problem.simulation.printState()
                for op in problem.operators:
                    op.apply(problem.simulation)
                problem.simulation.advance()
                self.theMainLoop.emit(QtCore.SIGNAL("step()"))
        ## Object handling main loop in a secondary thread
        self.theMainLoop = MainLoop(problem_step)
        ## Secondary thread
        self.thread = QtCore.QThread()
        self.theMainLoop.moveToThread(self.thread)
        QtCore.QObject.connect(self.theMainLoop,
                               QtCore.SIGNAL("step()"),
                               self.theMainLoop.step)


class MainLoop(QtCore.QObject):
    """
    Object that handle steps of the main computational loop.
    """

    def __init__(self, function):
        """
        Set the step function

        @param function : the step function
        """
        super(MainLoop, self).__init__()
        self.function = function

    def step(self):
        """Call the step function"""
        self.function()


class TestWindow(QtGui.QMainWindow):
    """
    Window definiton.

    This window contains a central widget which is the QGLWidget displaying
    OpenGL buffers and a StatusBar containing simulation informations.
    """

    def __init__(self):
        super(TestWindow, self).__init__()
        self.widget = GLWidget()
        self.setGeometry(100, 100, self.widget.width, self.widget.height)
        self.setCentralWidget(self.widget)
        self.label = QtGui.QLabel("")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.label)
        self.show()


class GLWidget(QGLWidget):
    """
    Qt widget that display OpenGL content.
    """
    def __init__(self):
        super(GLWidget, self).__init__()
        self.gl_objects = None
        self.color_vbo, self.pos_vbo = None, None
        self.partNumber = None
        self.width, self.height = 600, 600

    def setup(self, gl_buffers, color_vbo, pos_vbo, partNumber):
        """Set up VBOs and GLBuffers"""
        self.gl_objects = gl_buffers
        self.color_vbo, self.pos_vbo = color_vbo, pos_vbo
        self.partNumber = partNumber

    @debug
    def initializeGL(self):
        """GL content initialization"""
        self.buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, npw.zeros(2),
                        gl.GL_DYNAMIC_DRAW)
        self.cl_env = get_opengl_shared_environment(
            platform_id=0, device_id=0,device_type='gpu',
            precision=HYSOP_REAL, resolution=None)

    @debug
    def paintGL(self):
        """Drawing function"""
        if not self.gl_objects is None:
            cl.enqueue_release_gl_objects(self.cl_env.queue, self.gl_objects)
            self.cl_env.queue.finish()
            # OpenGL draw
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.color_vbo)
            gl.glColorPointer(4, gl.GL_FLOAT, 0, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pos_vbo)
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.partNumber)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            gl.glFlush()
            cl.enqueue_acquire_gl_objects(self.cl_env.queue, self.gl_objects)

    @debug
    def resizeGL(self, width, height):
        """Call on resizing the window"""
        self.width, self.height = width, height
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, 0, 1)

