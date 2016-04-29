/**
 * @file rendering.cl
 * Rendering kernels.
 */

/**
 * Colorize regarding scalar values.
 *
 * @param scalar Scalar values used
 * @param color Color data array that contains RGBA values for each grid point
 */
__kernel void colorize(__global const float* scalar,
		       __global float* color
)
{
  __private uint ind;
  __private float c;
  __private int ix, iy;
  ix = get_global_id(0);
  iy = get_global_id(1);
  ind = ix + iy*NB_X;

  //plain colors
  /* c = (scalar[ind] > 0.5f ? 1.0: 0.0); */
  /* color[4*ind + 0] = c; //Red */
  /* color[4*ind + 1] = 0.0; //Green */
  /* color[4*ind + 2] = 0.0; //Blue */
  /* color[4*ind + 3] = 1.0; //Alpha */

  //shaded colors
  c = scalar[ind];
  color[4*ind + 0] = 2.0*c; //Red
  color[4*ind + 1] = 2.0*c-0.5; //Green
  color[4*ind + 2] = 2.0*c-1.0; //Blue
  color[4*ind + 3] = 1.0; //Alpha
}


/**
 * Compute grid point coordinates from OpenCL index space.
 *
 * @param pos Coordinates ax XY values for each grid point.
 * @param minPos Domain origin.
 * @param size Mesh size.
 */
__kernel void initPointCoordinates(__global float* pos, float4 minPos, float4 size)
{
  __private uint ind;
  __private int ix, iy;
  ix = get_global_id(0);
  iy = get_global_id(1);
  ind = ix + iy*NB_X;

  pos[2*ind + 0] = ix*size.x;
  pos[2*ind + 1] = iy*size.y;
}

