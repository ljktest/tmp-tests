/**
 * @file transpose_xy.cl
 * Transposition in XY plane, coalesced, diagonal coordinates, vectorized version.
 */

/**
 * Performs a transposition in xy plane.
 * Optimizations used are:
 *   - Coalesced reads and writes by means of local memory buffer (tile),
 *   - Local memory padding to avoir banck conflicts (optional),
 *   - Work groups are mapped to diagonal coordinates in global memory,
 *   - Reads and writes are performed by OpenCL vector types.
 *
 * A work group handle transposition for a tile. Transposition is done when reading data in tile.
 * Work-group layout: \code
 * ________________________
 * |0,0 | 1,0 | ...
 * |N,0 | 0,1 | 1,2 | ...
 * | .     .  | 0,2 | ...
 * | .     .
 * | .     .
 * |
 * \endcode
 *
 * @param in Input data
 * @param out Output data
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last. Output layout is <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code>.
 * @remark <code>PADDING_XY</code> : local memory padding width.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @see hysop.gpu.tools.parse_file
 */
__kernel void transpose_xy(__global const float* in,
			   __global float* out)
{
  float__N__ temp;			/* Temporary variable */
  uint group_id_x;			/* Work-group coordinate in global space index X */
  uint group_id_y;			/* Work-group coordinate in global space index Y */
  uint lid_x = get_local_id(0);
  uint lid_y = get_local_id(1);

  uint xIndex, yIndex, zIndex;
  uint index_in, index_out;
  uint gidI, gidII, i;

  __local float tile[TILE_DIM_XY][TILE_DIM_XY+PADDING_XY]; /* Tile with padding */

#ifdef NB_III
  for(zIndex=get_global_id(2); zIndex<NB_III; zIndex+=get_global_size(2))
#else
    zIndex=get_global_id(2);
#endif
  {
    for(gidI=get_group_id(0); gidI<NB_GROUPS_I; gidI+=get_num_groups(0)) {
      for(gidII=get_group_id(1); gidII<NB_GROUPS_II; gidII+=get_num_groups(1)) {

	/* Use of diagonal coordinates */
#if NB_II == NB_I
	group_id_x = (gidI + gidII) % NB_GROUPS_I;
	group_id_y = gidI;
#else
	uint bid = gidI + gidII * NB_GROUPS_I;
	group_id_y = bid%NB_GROUPS_II;
	group_id_x = ((bid/NB_GROUPS_II) + group_id_y)%NB_GROUPS_I;
#endif

	/* Global input index for work-item */
	xIndex = group_id_x * TILE_DIM_XY + lid_x*__N__;
	yIndex = group_id_y * TILE_DIM_XY + lid_y;
	//zIndex = get_global_id(2);
	index_in = xIndex + yIndex * NB_II + zIndex * NB_II * NB_I;

	/* Global output index */
	xIndex = group_id_y * TILE_DIM_XY + lid_x*__N__;
	yIndex = group_id_x * TILE_DIM_XY + lid_y;
	index_out = xIndex + yIndex * NB_I + zIndex * NB_I * NB_II;


	for(i=0; i<TILE_DIM_XY; i+=BLOCK_ROWS_XY) {
	  /* Fill the tile */
	  temp = vload__N__((index_in + i * NB_II)/__N__, in);
	  tile[lid_y + i][lid_x*__N__+__NN__] = temp.s__NN__;
	}

	/* Synchronize work-group */
	barrier(CLK_LOCAL_MEM_FENCE);

	for(i=0; i<TILE_DIM_XY; i+=BLOCK_ROWS_XY) {
	  /* Write transposed data */
	  temp = (float__N__)(tile[lid_x*__N__+__NN__][lid_y + i],
			      );
	  vstore__N__(temp, (index_out + i*NB_I)/__N__, out);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
}
