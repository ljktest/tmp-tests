/**
 * @file advection_and_remeshing.cl
 * Advection and remeshing kernel, vectorized version.
 */

/**
 * Performs advection and then remeshing of the particles scalar.
 * A work-group is handling a 1D problem. Thus, gidY and gidZ are constants among work-items of a work-group. Computations of 1D problems are placed in loops over gidY and gidZ to adjust local workload and handle the work-item maximum size.
 * Each work-item computes NB_I/WI_NB particles positions. To avoid concurrent witings, in case of strong velocity gradients, work-items computes contiguous particles.
 * Particle are computed through OpenCL vector types of lenght 2, 4 or 8.
 * Scalar results are stored in a local buffer as a cache and then copied to global memory buffer.
 *
 * @param gvelo Velocity field
 * @param pscal Particle scalar
 * @param gscal Grid scalar
 * @param dt Time step
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
 * @remark <code>__RCOMP_I</code> flag is for instruction expansion for the different remeshed components.
 * @remark <code>__RCOMP_P</code> flag is for function parameter expansion for the different remeshed components.
 * @remark <code>__ID__</code> is replaced by the remeshed component id in an expansion.
 * @see hysop.gpu.tools.parse_file
 */
__kernel void advection_and_remeshing(__global const float* gvelo,
				      __RCOMP_P__global const float* pscal__ID__,
				      __RCOMP_P__global float* gscal__ID__,
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
  float__N__ p;			/* Particle position */
  __RCOMP_I float__N__ s__ID__; /* Particle scalar */
  uint line_index; /* Current 1D problem index */

  __RCOMP_I__local float gscal_loc__ID__[NB_I]; /* Local buffer for result */
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

      for(i=gidX*__N__; i<NB_I; i+=(WI_NB*__N__)) {
	/* Initialize result buffer */
	__RCOMP_Igscal_loc__ID__[noBC_id(i+__NN__)] = 0.0;
      }

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=gidX*PART_NB_PER_WI; i<(gidX + 1)*PART_NB_PER_WI; i+=__N__) {
	/* Read Particle scalar */
	__RCOMP_Is__ID__ = vload__N__((i + line_index)/__N__, pscal__ID__);
	/* Compute particle position */
	p = advection(i, dt, velocity_cache, mesh);
	/* Remesh particle */
	remesh(i, __RCOMP_Ps__ID__, p, __RCOMP_Pgscal_loc__ID__, mesh);
      }

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=gidX*__N__; i<NB_I; i+=(WI_NB*__N__)) {
	/* Store result */
	__RCOMP_Ivstore__N__((float__N__)(gscal_loc__ID__[noBC_id(i+__NN__)],
					  ), (i + line_index)/__N__, gscal__ID__);
      }

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
