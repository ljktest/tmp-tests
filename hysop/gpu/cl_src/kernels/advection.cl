/**
 * @file advection.cl
 * Advection kernel, vectorized version.
 */

/**
 * Computes particles positions from the velocity field.
 * A work-group is handling a 1D problem. Thus, gidY and gidZ are constants among work-items of a work-group. Computations of 1D problems are placed in loops over gidY and gidZ to adjust local workload and handle the work-item maximum size.
 * Each work-item computes <code>NB_I/WI_NB</code> particles positions in each 1D problem.
 * Particle are computed through OpenCL vector types of length 2, 4 or 8.
 * Velocity data are copied to a local buffer as a cache.
 *
 * @param gvelo Velocity.
 * @param ppos Particle position.
 * @param dt Time step.
 * @param mesh Mesh description.
 * @param inv_v_dx_y velocity grid 1/dy
 * @param inv_v_dx_z velocity grid 1/dz
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last.
 * @remark <code>NB_X</code>, <code>NB_Y</code>, <code>NB_Z</code> : points number in physical space directions.
 * @remark <code>WI_NB</code> corresponds to the work-item number.
 * @remark <code>ADVEC_IS_MULTISCALE</code> is a flag for multiscale.
 * @remark <code>V_NB_I</code>, <code>V_NB_II</code>, <code>V_NB_III</code> : points number for velocity grid in directions from 1st varying index to last.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @see hysop.gpu.tools.parse_file
 */
__kernel void advection_kernel(__global const float* gvelo,
			       __global float* ppos,
			       float dt,
#if ADVEC_IS_MULTISCALE
			       float inv_v_dx_y, float inv_v_dx_z,
#endif
			       __constant struct AdvectionMeshInfo* mesh)
{
  uint gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  uint gidY; /* OpenCL work-itme global index (Y) */
  uint gidZ; /* OpenCL work-itme global index (Z) */
  uint i;			/* Particle index in 1D problem */
  float__N__ p;				/* Particle position */
  uint line_index; /* Current 1D problem index */

  __local float velocity_cache[V_NB_I]; /* Velocity cache */

  for(gidZ=get_global_id(2);
#ifdef NB_Z
      gidZ<NB_III;
#else
      gidZ<=get_global_id(2); // Single element loop
#endif
      gidZ+=get_global_size(2)) {
    for(gidY=get_global_id(1); gidY<NB_II; gidY+=get_global_size(1)) {

      // 1D problem computations
      line_index = gidY*NB_I+ gidZ*NB_I*NB_II;

#if ADVEC_IS_MULTISCALE
      fill_velocity_cache(gvelo, gidX, gidY, gidZ, velocity_cache, inv_v_dx_y, inv_v_dx_z, mesh);
#else
      fill_velocity_cache(gvelo, gidX, gidY, gidZ, velocity_cache, mesh);
#endif

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=gidX*__N__; i<NB_I; i+=WI_NB*__N__) {
	/* Compute position */
	p = advection(i, dt, velocity_cache, mesh);
	/* Store result */
	vstore__N__(p, (i+line_index)/__N__, ppos);
      }

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
