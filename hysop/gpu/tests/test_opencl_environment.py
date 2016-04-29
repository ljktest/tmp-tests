from hysop.constants import np
from hysop.gpu.tools import get_opencl_environment
FLOAT_GPU = np.float32


def test_queue_unique_creation():
    """
    Testing that only one queue is created when multiples calls to get
    an environment.
    """
    cl_env = get_opencl_environment()
    cl_env_id = id(cl_env)
    cl_envb = get_opencl_environment()
    cl_envb_id = id(cl_envb)
    assert cl_env_id == cl_envb_id


def test_parse_src_expand_floatN():
    """
    """
    import StringIO
    cl_env = get_opencl_environment()
    str_as_src = """
    vstore__N__((float__N__)(gscal_loc[noBC_id(i+__NN__,nb_part)],
    ), (i + gidY*WIDTH)/__N__, gscal);
    """
    parsed_str_as_src = """
    vstore4((float4)(gscal_loc[noBC_id(i+0,nb_part)],""" + \
        """gscal_loc[noBC_id(i+1,nb_part)],""" + \
        """gscal_loc[noBC_id(i+2,nb_part)],gscal_loc[noBC_id(i+3,nb_part)]
    ), (i + gidY*WIDTH)/4, gscal);
    """
    buf = StringIO.StringIO(str_as_src)
    res = cl_env.parse_file(buf, n=4)
    assert len(parsed_str_as_src) == len(res)
    for s1, s2 in zip(parsed_str_as_src, res):
        assert s1 == s2


def test_parse_src_expand():
    """
    """
    import StringIO
    cl_env = get_opencl_environment()
    str_as_src = """
    gvelo_loc[noBC_id(i+__NN__,nb_part)] = v.s__NN__;
    """
    parsed_str_as_src = """
    gvelo_loc[noBC_id(i+0,nb_part)] = v.s0;
    gvelo_loc[noBC_id(i+1,nb_part)] = v.s1;
    gvelo_loc[noBC_id(i+2,nb_part)] = v.s2;
    gvelo_loc[noBC_id(i+3,nb_part)] = v.s3;
    """
    buf = StringIO.StringIO(str_as_src)
    res = cl_env.parse_file(buf, n=4)
    assert len(parsed_str_as_src) == len(res)
    for s1, s2 in zip(parsed_str_as_src, res):
        assert s1 == s2


def test_parse_expand_remeshed_component():
    """
    """
    import StringIO
    cl_env = get_opencl_environment()
    str_as_src = """
    __kernel void advection_and_remeshing(__global const float* gvelo,
                      __RCOMP_P__global const float* pscal__ID__,
                      __RCOMP_P__global float* gscal__ID__,
                      __local float* gvelo_loc,
                      __RCOMP_P__local float* gscal_loc__ID__,
                      float dt,float min_position, float dx)
    {
     __RCOMP_I gscal_loc__ID__[noBC_id(i)] = 0.0;
      remesh(i, dx, invdx, s, p, __RCOMP_Pgscal_loc__ID__);
      test(__RCOMP_Pgscal_loc__ID__, __RCOMP_Ppscal__ID__);
     __RCOMP_I gscal__ID__[i + line_index] = gscal_loc__ID__[noBC_id(i)];
     __RCOMP_I vstore__N__((float__N__)(gscal_loc__ID__[noBC_id(i+__NN__)],
                   ), (i + line_index)/__N__, gscal__ID__);

    """
    parsed_str_as_src_2components = """
    __kernel void advection_and_remeshing(__global const float* gvelo,
                      """ + \
        """__global const float* pscal0, __global const float* pscal1,
                      __global float* gscal0, __global float* gscal1,
                      __local float* gvelo_loc,
                      __local float* gscal_loc0, __local float* gscal_loc1,
                      float dt,float min_position, float dx)
    {
      gscal_loc0[noBC_id(i)] = 0.0; gscal_loc1[noBC_id(i)] = 0.0;
      remesh(i, dx, invdx, s, p, gscal_loc0, gscal_loc1);
      test(gscal_loc0, gscal_loc1, pscal0, pscal1);
      gscal0[i + line_index] = gscal_loc0[noBC_id(i)]; """ + \
        """gscal1[i + line_index] = gscal_loc1[noBC_id(i)];
      vstore4((float4)(gscal_loc0[noBC_id(i+0)],""" + \
        """gscal_loc0[noBC_id(i+1)],gscal_loc0[noBC_id(i+2)],""" + \
        """gscal_loc0[noBC_id(i+3)]
                   ), (i + line_index)/4, gscal0); """ + \
        """vstore4((float4)(gscal_loc1[noBC_id(i+0)],""" + \
        """gscal_loc1[noBC_id(i+1)],gscal_loc1[noBC_id(i+2)],""" + \
        """gscal_loc1[noBC_id(i+3)]
                   ), (i + line_index)/4, gscal1);

    """
    parsed_str_as_src_1components = """
    __kernel void advection_and_remeshing(__global const float* gvelo,
                      __global const float* pscal0,
                      __global float* gscal0,
                      __local float* gvelo_loc,
                      __local float* gscal_loc0,
                      float dt,float min_position, float dx)
    {
      gscal_loc0[noBC_id(i)] = 0.0;
      remesh(i, dx, invdx, s, p, gscal_loc0);
      test(gscal_loc0, pscal0);
      gscal0[i + line_index] = gscal_loc0[noBC_id(i)];
      vstore4((float4)(gscal_loc0[noBC_id(i+0)],""" + \
        """gscal_loc0[noBC_id(i+1)],gscal_loc0[noBC_id(i+2)],""" + \
        """gscal_loc0[noBC_id(i+3)]
                   ), (i + line_index)/4, gscal0);

    """
    buf = StringIO.StringIO(str_as_src)
    res = cl_env.parse_file(buf, n=4, nb_remesh_components=1)
    print res
    assert len(parsed_str_as_src_1components) == len(res)
    for s1, s2 in zip(parsed_str_as_src_1components, res):
        assert s1 == s2

    buf = StringIO.StringIO(str_as_src)
    res = cl_env.parse_file(buf, n=4, nb_remesh_components=2)
    print res
    assert len(parsed_str_as_src_2components) == len(res)
    for s1, s2 in zip(parsed_str_as_src_2components, res):
        assert s1 == s2
