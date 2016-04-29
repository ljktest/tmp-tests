/**
 * @file advection_and_remeshing.cl
 * Euler advection and remeshing kernel.
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
 * @param min_position Domain lower coordinate
 * @param dx Space step
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 * @remark WI_NB corresponds to the work-item number.
 * @remark \__N__ is expanded at compilation time by vector width.
 * @remark \__NN__ is expanded at compilation time by a sequence of integer for each vector component.
 * @remark <code>__RCOMP_I</code> flag is for instruction expansion for the different remeshed components.
 * @remark <code>__RCOMP_P</code> flag is for function parameter expansion for the different remeshed components.
 * @remark <code>__ID__</code> is replaced by the remeshed component id in an expansion.
 * @see hysop.gpu.tools.parse_file
 */
__kernel void advection_and_remeshing(__global const float* gvelo,
				      __RCOMP_P__global const float* pscal__ID__,
				      __RCOMP_P__global float* gscal__ID__,
				      float dt,
				      __constant struct AdvectionMeshInfo* mesh)
{
  uint gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  uint gidY; /* OpenCL work-itme global index (Y) */
  uint gidZ; /* OpenCL work-itme global index (Z) */
  uint i;			/* Particle index in 1D problem */
  float p,c;			/* Particle position */
  __RCOMP_I float s__ID__;	/* Particle scalar */
  uint line_index; /* Current 1D problem index */

  __RCOMP_I__local float gscal_loc__ID__[NB_I]; /* Local buffer for result */

  for(gidZ=get_global_id(2);
#ifdef NB_Z
      gidZ<NB_III;
#else
      gidZ<=get_global_id(2); // Single element loop
#endif
      gidZ+=get_global_size(2)) {
    for(gidY=get_global_id(1); gidY<NB_II; gidY+=get_global_size(1)) {

      // 1D computations
      line_index = gidY*NB_I+ gidZ*NB_I*NB_II;

      for(i=gidX; i<NB_I; i+=(WI_NB)) {
	/* Initialize result buffer */
	__RCOMP_Igscal_loc__ID__[noBC_id(i)] = 0.0;
      }

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=gidX*PART_NB_PER_WI; i<(gidX + 1)*PART_NB_PER_WI; i+=1) {
	/* Read Particle scalar */
	__RCOMP_Is__ID__ = pscal__ID__[i + line_index];
	/* Compute particle position */
	c = fma(i, mesh->dx.x, mesh->min_position);
	p = fma(dt, gvelo[i+line_index], c);
	/* Remesh particle */
	remesh(i, __RCOMP_Ps__ID__, p, __RCOMP_Pgscal_loc__ID__, mesh);
      }

      /* Synchronize work-group */
      barrier(CLK_LOCAL_MEM_FENCE);

      for(i=gidX; i<NB_I; i+=(WI_NB)) {
	/* Store result */
	__RCOMP_Igscal__ID__[i + line_index] = gscal_loc__ID__[noBC_id(i)];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
