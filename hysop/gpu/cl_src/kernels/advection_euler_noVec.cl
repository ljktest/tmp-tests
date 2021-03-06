/**
 * @file advection_euler_noVec.cl
 * Advection kernel, basic version for Euler integrator for simple scale problems (no need velocity cache).
 */

/**
 * Computes particles positions from the velocity field.
 * A work-group is handling a 1D problem. Thus, gidY and gidZ are constants among work-items of a work-group.
 * Each work-item computes NB_I/WI_NB particles positions.
 *
 * @param gvelo Velocity.
 * @param ppos Particle position.
 * @param dt Time step.
 * @param min_position Domain lower coordinate.
 * @param dx Space step.
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 * @remark WI_NB corresponds to the work-item number.
 */
__kernel void advection_kernel(__global const float* gvelo,
			       __global float* ppos,
			       float dt,
			       __constant struct AdvectionMeshInfo* mesh)
{
  uint gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  uint gidY; /* OpenCL work-itme global index (Y) */
  uint gidZ; /* OpenCL work-itme global index (Z) */
  uint i;			/* Particle index in 1D problem */
  uint line_index; /* Current 1D problem index */
  float c;

  for(gidZ=get_global_id(2);
#ifdef NB_Z
      gidZ<NB_III;
#else
      gidZ<=get_global_id(2); // Single element loop
#endif
      gidZ+=get_global_size(2)) {
    for(gidY=get_global_id(1); gidY<NB_II; gidY+=get_global_size(1)) {

      //1D computations
      line_index = gidY*NB_I+ gidZ*NB_I*NB_II;

      for(i=gidX; i<NB_I; i+=WI_NB) {
	c = fma(i, mesh->dx.x, mesh->min_position);
	ppos[i+line_index] =  fma(dt, gvelo[i+line_index], c);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
