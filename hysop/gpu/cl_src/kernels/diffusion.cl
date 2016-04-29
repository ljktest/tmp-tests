/**
 * @file diffusion.cl
 * Diffusion kernel.
 */

/**
 * Computes diffusion operator with finite differences.
 * Stencil computation is performed within a 2D index space of size <code>TILE_SIZE</code> by a work-group. The 3rd direction is traversed in a loop for data reuse.
 *
 * @param scal_in Input scalar field
 * @param ghostsX Ghosts array if X is a communication direction
 * @param ghostsY Ghosts array if Y is a communication direction
 * @param ghostsZ Ghosts array if Z is a communication direction
 * @param scal_out Output scalar field
 * @param nudt Diffusion coefficient
 * @param dx Mesh space step
 *
 * @remark <code>NB_X</code>, <code>NB_Y</code>, <code>NB_Z</code> : points number in physical space directions.
 * @remark <code>NB_PART</code> Particles number per work-item in computing direction
 * @remark <code>CUT_DIT_X</code>, <code>CUT_DIT_Y</code> and <code>CUT_DIT_Z</code> : flags for communication direction
 * @remark <code>NB_GROUPS_I</code> and <code>NB_GROUPS_II</code> : tiles number in X and Y directions.
 * @remark <code>L_WIDTH</code> : work-item number in tile.
 */

__kernel void diffusion(__global const float* scal_in,
#if CUT_DIR_X == 1
			__global const float* ghostsX,
#endif
#if CUT_DIR_Y == 1
			__global const float* ghostsY,
#endif
#if CUT_DIR_Z == 1
			__global const float* ghostsZ,
#endif
			__global float* scal_out,
			float nudt,
			float4 dx)
{
  int t_gidX, t_gidY;
  int lidX, lidY;
  int gidX, gidY, gidZ;
  float cx, cy, cz;
  float scal_z_m[NB_PART];
  float scal_z[NB_PART];
  float scal_z_p[NB_PART];
  float s;
  uint i;

  __local float tile_XY[TILE_SIZE+2][TILE_SIZE+2];

  for (t_gidX=get_group_id(0); t_gidX<NB_GROUPS_I; t_gidX+=get_num_groups(0)) {
    for (t_gidY=get_group_id(1); t_gidY<NB_GROUPS_II; t_gidY+=get_num_groups(1)) {

      // Tile computation
      lidX = get_local_id(0);
      lidY = get_local_id(1);
      gidX = t_gidX*TILE_SIZE + lidX; /* OpenCL work-item global index (X) */
      gidY = t_gidY*TILE_SIZE + lidY; /* OpenCL work-item global index (Y) */
      cx = nudt/(dx.x*dx.x);
      cy = nudt/(dx.y*dx.y);
      cz = nudt/(dx.z*dx.z);

      for(i=0;i<NB_PART;i++) {
#if CUT_DIR_Z == 1
	scal_z_m[i] = ghostsZ[gidX + (gidY+i*L_WIDTH)*NB_X + NB_X*NB_Y];
#else
	scal_z_m[i] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X + (NB_Z-1)*NB_X*NB_Y];
#endif
	scal_z[i] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X];
      }

      lidX += 1;
      lidY += 1;

      // loop over Z indices but last.
      for (gidZ=0; gidZ<(NB_Z-1); gidZ++) {
	for(i=0;i<NB_PART;i++) {
	  // fill the tile
	  tile_XY[lidX][lidY+i*L_WIDTH] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];

	  /* // fill tile edges */
#if CUT_DIR_X == 1
	  tile_XY[0][lidY+i*L_WIDTH] = (t_gidX*TILE_SIZE>=1) ? scal_in[t_gidX*TILE_SIZE-1 + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y] : ghostsX[1 + (gidY+i*L_WIDTH)*2 + gidZ*2*NB_Y];
	  tile_XY[TILE_SIZE+1][lidY+i*L_WIDTH] = ((t_gidX+1)*TILE_SIZE<NB_X) ? scal_in[(t_gidX+1)*TILE_SIZE + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y]: ghostsX[(gidY+i*L_WIDTH)*2 + gidZ*2*NB_Y];
#else
	  tile_XY[0][lidY+i*L_WIDTH] = scal_in[((t_gidX*TILE_SIZE-1+NB_X)%NB_X) + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];
	  tile_XY[TILE_SIZE+1][lidY+i*L_WIDTH] = scal_in[(((t_gidX+1)*TILE_SIZE+NB_X)%NB_X) + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];
#endif
	}
#if CUT_DIR_Y == 1
	tile_XY[lidX][0] = (t_gidY*TILE_SIZE>=1)? scal_in[gidX + (t_gidY*TILE_SIZE-1)*NB_X + gidZ*NB_X*NB_Y] : ghostsY[gidX + NB_X + gidZ*NB_X*2];
	tile_XY[lidX][TILE_SIZE+1] = ((t_gidY+1)*TILE_SIZE<NB_Y) ? scal_in[gidX + (t_gidY+1)*TILE_SIZE*NB_X + gidZ*NB_X*NB_Y] : ghostsY[gidX + gidZ*NB_X*2];
#else
	tile_XY[lidX][0] = scal_in[gidX + ((t_gidY*TILE_SIZE-1+NB_Y)%NB_Y)*NB_X + gidZ*NB_X*NB_Y];
	tile_XY[lidX][TILE_SIZE+1] = scal_in[gidX + (((t_gidY+1)*TILE_SIZE+NB_Y)%NB_Y)*NB_X + gidZ*NB_X*NB_Y];
#endif

	/* Synchronize work-group */
	barrier(CLK_LOCAL_MEM_FENCE);

	for(i=0;i<NB_PART;i++) {
	  /* get scalar value in Z direction */
	  scal_z_p[i] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X + (gidZ+1)*NB_X*NB_Y];

	  // Compute stencil
	  // central point
	  s = scal_z[i] * (1.0 - 2.0 * (cx + cy + cz));

	  s += cz*(scal_z_m[i] + scal_z_p[i]);

	  s += cy * tile_XY[lidX][lidY+i*L_WIDTH-1];
	  s += cy * tile_XY[lidX][lidY+i*L_WIDTH+1];
	  s += cx * tile_XY[lidX-1][lidY+i*L_WIDTH];
	  s += cx * tile_XY[lidX+1][lidY+i*L_WIDTH];

	  // write result
	  scal_out[gidX + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y] = s;
	}

	/* Synchronize work-group */
	barrier(CLK_LOCAL_MEM_FENCE);

	for(i=0;i<NB_PART;i++) {
	  // Shift Z values
	  scal_z_m[i] = scal_z[i];
	  scal_z[i] = scal_z_p[i];
	}
      }

      // Compute last point (from ghosts)
      gidZ = NB_Z - 1;

      for(i=0;i<NB_PART;i++) {
	// fill the tile
	tile_XY[lidX][lidY+i*L_WIDTH] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];

	/* // fill tile edges */
#if CUT_DIR_X == 1
	tile_XY[0][lidY+i*L_WIDTH] = (t_gidX*TILE_SIZE>=1) ? scal_in[t_gidX*TILE_SIZE-1 + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y] : ghostsX[1 + (gidY+i*L_WIDTH)*2 + gidZ*2*NB_Y];
	tile_XY[TILE_SIZE+1][lidY+i*L_WIDTH] = ((t_gidX+1)*TILE_SIZE<NB_X) ? scal_in[(t_gidX+1)*TILE_SIZE + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y]: ghostsX[(gidY+i*L_WIDTH)*2 + gidZ*2*NB_Y];
#else
	tile_XY[0][lidY+i*L_WIDTH] = scal_in[((t_gidX*TILE_SIZE-1+NB_X)%NB_X) + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];
	tile_XY[TILE_SIZE+1][lidY+i*L_WIDTH] = scal_in[(((t_gidX+1)*TILE_SIZE+NB_X)%NB_X) + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y];
#endif
      }
#if CUT_DIR_Y == 1
      tile_XY[lidX][0] = (t_gidY*TILE_SIZE>=1)? scal_in[gidX + (t_gidY*TILE_SIZE-1)*NB_X + gidZ*NB_X*NB_Y] : ghostsY[gidX + NB_X + gidZ*NB_X*2];
      tile_XY[lidX][TILE_SIZE+1] = ((t_gidY+1)*TILE_SIZE<NB_Y) ? scal_in[gidX + (t_gidY+1)*TILE_SIZE*NB_X + gidZ*NB_X*NB_Y] : ghostsY[gidX + gidZ*NB_X*2];
#else
      tile_XY[lidX][0] = scal_in[gidX + ((t_gidY*TILE_SIZE-1+NB_Y)%NB_Y)*NB_X + gidZ*NB_X*NB_Y];
      tile_XY[lidX][TILE_SIZE+1] = scal_in[gidX + (((t_gidY+1)*TILE_SIZE+NB_Y)%NB_Y)*NB_X + gidZ*NB_X*NB_Y];
#endif

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=0;i<NB_PART;i++) {
	/* // get scalar value in Z direction */
#if CUT_DIR_Z == 1
	scal_z_p[i] = ghostsZ[gidX + (gidY+i*L_WIDTH)*NB_X];
#else
	scal_z_p[i] = scal_in[gidX + (gidY+i*L_WIDTH)*NB_X];
#endif

	// Compute stencil
	/* // central point */
	s = scal_z[i] * (1.0 - 2.0 * (cx + cy + cz));

	s += cz*(scal_z_m[i] + scal_z_p[i]);

	s += cy * tile_XY[lidX][lidY+i*L_WIDTH-1];
	s += cy * tile_XY[lidX][lidY+i*L_WIDTH+1];
	s += cx * tile_XY[lidX-1][lidY+i*L_WIDTH];
	s += cx * tile_XY[lidX+1][lidY+i*L_WIDTH];

	// write result
	scal_out[gidX + (gidY+i*L_WIDTH)*NB_X + gidZ*NB_X*NB_Y] = s;
      }
      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
