/**
 * @file transpose_xz.cl
 * Transposition in XZ plane, coalesced, diagonal coordinates, 3D tiles.
 */

/**
 * Perfoms a transposition in XZ plane. As data have to be contiguously read an write in global memory, we use a 3D tile.
 * Optimizations used are:
 *   - Coalesced reads and writes by means of local memory buffer (tile),
 *   - Local memory padding to avoir banck conflicts (optional),
 *   - Work groups are mapped to diagonal coordinates in global memory,
 *   - Reads and writes are performed by OpenCL vector types.
 *
 *
 * @param in Input data
 * @param out Output data
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last. Output layout is <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code>.
 * @remark <code>PADDING_XZ</code> : local memory padding width.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @see hysop.gpu.tools.parse_file
 * @see transpose_xy.cl
 */
__kernel void transpose_xz(__global const float* in,
			   __global float* out)
{
  float__N__ temp;			/* Temporary variable */
  uint group_id_x;			/* Work-group coordinate in global space index X */
  uint group_id_z;			/* Work-group coordinate in global space index Y */
  uint lid_x = get_local_id(0);
  uint lid_z = get_local_id(2);

  uint xIndex, yIndex, zIndex;
  uint index_in, index_out;
  uint gidI, gidIII, j;

  __local float tile[TILE_DIM_XZ][TILE_DIM_XZ+PADDING_XZ]; /* Tile with padding */

  for(yIndex=get_global_id(1); yIndex<NB_II; yIndex+=get_global_size(1)) {
    for(gidI=get_group_id(0); gidI<NB_GROUPS_I; gidI+=get_num_groups(0)) {
      for(gidIII=get_group_id(2); gidIII<NB_GROUPS_III; gidIII+=get_num_groups(2)) {

	/* Use of diagonal coordinates */
#if NB_III == NB_I
	group_id_x = (gidI + gidIII) % NB_GROUPS_I;
	group_id_z = gidI;
#else
	uint bid = gidI + gidIII * NB_GROUPS_I;
	group_id_z = bid%NB_GROUPS_III;
	group_id_x = ((bid/NB_GROUPS_III) + group_id_z)%NB_GROUPS_I;
#endif

	/* Global input index for work-item */
	xIndex = group_id_x * TILE_DIM_XZ + lid_x*__N__;
	zIndex = group_id_z * TILE_DIM_XZ + lid_z;
	index_in = xIndex + yIndex * NB_III + zIndex * NB_III * NB_II;

	/* Global output index */
	xIndex = group_id_z * TILE_DIM_XZ + lid_x*__N__;
	zIndex = group_id_x * TILE_DIM_XZ + lid_z;
	index_out = xIndex + yIndex * NB_I + zIndex * NB_I * NB_II;

	for(j=0; j<TILE_DIM_XZ; j+=BLOCK_DEPH_XZ) {
	  /* Fill the tile */
	  temp = vload__N__((index_in + j*NB_III*NB_II)/__N__, in);
	  tile[lid_z + j][lid_x*__N__+__NN__] = temp.s__NN__;

	}
	/* Synchronize work-group */
	barrier(CLK_LOCAL_MEM_FENCE);

	for(j=0; j<TILE_DIM_XZ; j+=BLOCK_DEPH_XZ) {
	  /* Write transposed data */
	  temp = (float__N__)(tile[lid_x*__N__+__NN__][lid_z + j],
			      );
	  vstore__N__(temp, (index_out + j*NB_I*NB_II)/__N__, out);
	}
      }
    }
  }
}
