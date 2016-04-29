/**
 * @file copy_locMem.cl
 * Copy kernel, use local memory.
 */

/**
 * Performs a copy from in to out. Data are moved to local memory buffer.
 *
 * @param in Input data.
 * @param out Output data
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 */
__kernel void copy(__global const float* in,
		   __global float* out)
{
  uint xIndex = get_group_id(0) * TILE_DIM_COPY + get_local_id(0);
  uint yIndex = get_group_id(1) * TILE_DIM_COPY + get_local_id(1);
  uint zIndex = get_global_id(2);
  uint index = xIndex + yIndex * NB_I + zIndex*NB_I*NB_II;

  __local float tile[TILE_DIM_COPY][TILE_DIM_COPY];

  for(uint i=0; i<TILE_DIM_COPY; i+=BLOCK_ROWS_COPY)
    {
      tile[get_local_id(1)+i][get_local_id(0)] = in[index + i*NB_I];
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(uint i=0; i<TILE_DIM_COPY; i+=BLOCK_ROWS_COPY)
    {
      out[index + i*NB_I] = tile[get_local_id(1)+i][get_local_id(0)];
    }
}
