/**
 * @file copy.cl
 * Copy kernel, vectorized version.
 */

/**
 * Performs a copy from in to out. Data are read by blocs of <code>__N__</code> contiguously.
 *
 * @param in Input data.
 * @param out Output data
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last.
 */
__kernel void copy(__global const float* in,
		   __global float* out)
{
  uint xIndex = (get_group_id(0) * TILE_DIM_COPY + get_local_id(0)*__N__);
  uint yIndex = get_group_id(1) * TILE_DIM_COPY + get_local_id(1);
  uint zIndex = get_global_id(2);
  uint index = xIndex + yIndex * NB_I + zIndex*NB_I*NB_II;
  float x__NN__;

  for(uint i=0; i<TILE_DIM_COPY; i+=BLOCK_ROWS_COPY)
    {
      x__NN__ = in[index + __NN__ + i*NB_I];
      out[index + __NN__ + i*NB_I] = x__NN__;
    }
}
