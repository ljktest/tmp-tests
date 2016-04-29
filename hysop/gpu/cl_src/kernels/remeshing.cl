/**
 * @file remeshing.cl
 * Remeshing kernel.
 */

/**
 * Performs remeshing of the particles' scalar.
 * A work-group is handling a 1D problem. Thus, gidY and gidZ are constants among work-items of a work-group.
 * Each work-item computes <code>NB_I/WI_NB</code> particles positions. To avoid concurrent witings, in case of strong velocity gradients, work-items computes contiguous particles.
 * Particle are computed through OpenCL vector types of lenght 2, 4 or 8.
 * Scalar results are stored in a local buffer as a cache and then copied to global memory buffer.
 *
 * @param ppos Particle position
 * @param pscal Particle scalar
 * @param gscal Grid scalar
 * @param min_position Domain lower coordinate
 * @param dx Space step
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last.
 * @remark <code>WI_NB</code> corresponds to the work-item number.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @remark <code>__RCOMP_I</code> flag is for instruction expansion for the different remeshed components.
 * @remark <code>__RCOMP_P</code> flag is for function parameter expansion for the different remeshed components.
 * @remark <code>__ID__</code> is replaced by the remeshed component id in an expansion.
 * @see hysop.gpu.tools.parse_file
 */
__kernel void remeshing_kernel(__global const float* ppos,
			       __RCOMP_P__global const float* pscal__ID__,
			       __RCOMP_P__global float* gscal__ID__,
			       __constant struct AdvectionMeshInfo* mesh)
{
  uint gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  uint gidY; /* OpenCL work-itme global index (Y) */
  uint gidZ; /* OpenCL work-itme global index (Z) */
  //  float invdx = 1.0/dx;         /* Space step inverse */
  uint i;			/* Particle index in 1D problem */
  float__N__ p;			/* Particle position */
  __RCOMP_I float__N__ s__ID__; /* Particle scalar */
  uint line_index; /* Current 1D problem index */

  __RCOMP_I__local float gscal_loc__ID__[NB_I]; /* Local buffer for result */

#ifdef NB_Z
  for(gidZ=get_global_id(2); gidZ<NB_III; gidZ+=get_global_size(2)) {
#else
  gidZ=get_global_id(2); {
#endif
  for(gidY=get_global_id(1); gidY<NB_II; gidY+=get_global_size(1)) {
  line_index = gidY*NB_I+ gidZ*NB_I*NB_II;

  for(i=gidX*__N__; i<NB_I; i+=(WI_NB*__N__))
    {
      /* Initialize result buffer */
      __RCOMP_Igscal_loc__ID__[i+__NN__] = 0.0;
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=gidX*PART_NB_PER_WI; i<(gidX + 1)*PART_NB_PER_WI; i+=__N__)
    {
      /* Read particle position */
      p = vload__N__((i + line_index)/__N__, ppos);
      /* Read particle scalar */
      __RCOMP_Is__ID__ = vload__N__((i + line_index)/__N__, pscal__ID__);
      /* Remesh particle */
      remesh(i, __RCOMP_Ps__ID__, p, __RCOMP_Pgscal_loc__ID__, mesh);
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=gidX*__N__; i<NB_I; i+=(WI_NB*__N__))
    {
      /* Store result */
      __RCOMP_Ivstore__N__((float__N__)(gscal_loc__ID__[noBC_id(i+__NN__)],
			       ),(i + line_index)/__N__, gscal__ID__);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}
}
}
