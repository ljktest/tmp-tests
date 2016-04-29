/**
 * @file comm_remeshing_noVec.cl
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
 * @param buffer_l Buffer for storing out of domain contributions (to left)
 * @param buffer_r Buffer for storing out of domain contributions (to right)
 * @param min_position Domain lower coordinate
 * @param dx Space step
 * @param l_nb buffer_l sizes
 * @param r_nb buffer_r sizes
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
__kernel void buff_remesh_l(__global const float* ppos,
			    __global const float* pscal,
			    __global float* buffer_l,
			    int used_width,
			    __constant struct AdvectionMeshInfo* mesh
			    )
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  float p;			/* Particle position */
  float s;      /* Particle scalar */
float y;			/* Normalized distance to nearest left grid point */
  int ind;			/* Integer coordinate */
  int index;		/* Remeshing index */
  float w;

  uint line_index = gidY*NB_I+ gidZ*NB_I*NB_II; /* Current 1D problem index */

  float l_buff_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for(i=0; i<used_width; i++)
    l_buff_loc[i] = 0.0;

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  //for(i=lid*PART_NB_PER_WI; i<(lid + 1)*PART_NB_PER_WI; i+=1)
  for(i=0; i<2*BUFF_WIDTH; i++)
    {
      /* Read particle position */
      p = ppos[i + line_index];
      /* Read particle scalar */
      s = pscal[i + line_index];
      /* Remesh particle */

      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if(index<START_INDEX){ loc_ptr = l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store buffers
  for(i=0; i<used_width; i++)
    buffer_l[i + gidY*used_width + gidZ*used_width*NB_II] = l_buff_loc[i];
}

__kernel void buff_remesh_r(__global const float* ppos,
			    __global const float* pscal,
			    __global float* buffer_r,
			    int used_width,
			    __constant struct AdvectionMeshInfo* mesh
			    )
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  float p;			/* Particle position */
  float s;      /* Particle scalar */
  float y;			/* Normalized distance to nearest left grid point */
  int ind;			/* Integer coordinate */
  int index;		/* Remeshing index */
  float w;

  uint line_index = gidY*NB_I+ gidZ*NB_I*NB_II; /* Current 1D problem index */

  float r_buff_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for(i=0; i<used_width; i++)
    r_buff_loc[i] = 0.0;

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=NB_I-2*BUFF_WIDTH; i<NB_I; i++)
    {
      /* Read particle position */
      p = ppos[i + line_index];
      /* Read particle scalar */
      s = pscal[i + line_index];
      /* Remesh particle */

      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if(index > STOP_INDEX){ loc_ptr = loc_ptr = r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store buffers
  for(i=0; i<used_width; i++)
    buffer_r[i + gidY*used_width + gidZ*used_width*NB_II] = r_buff_loc[i];

}

__kernel void remesh(__global const float* ppos,
			  __global const float* pscal,
			  __global float* gscal,
			  __constant struct AdvectionMeshInfo* mesh
			  )
{
  int lid = get_local_id(0);	/* OpenCL work-itme global index (X) */
  int gidY = get_global_id(1); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(2); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  float p;			/* Particle position */
  float s;      /* Particle scalar */
float y;			/* Normalized distance to nearest left grid point */
  int ind;			/* Integer coordinate */
  int index;		/* Remeshing index */
  float w;

  uint line_index = gidY*NB_I+ gidZ*NB_I*NB_II; /* Current 1D problem index */

  __local float gscal_loc[NB_I];
  __local float* loc_ptr;

  /* Initialize result buffer */
  for(i=lid; i<NB_I; i+=WI_NB)
      gscal_loc[i] = 0.0;

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=lid*PART_NB_PER_WI; i<(lid + 1)*PART_NB_PER_WI; i+=1)
    {
      /* Read particle position */
      p = ppos[i + line_index];
      /* Read particle scalar */
      s = pscal[i + line_index];
      /* Remesh particle */

      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index>=START_INDEX && index <= STOP_INDEX) {loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Store result */
  for(i=lid; i<NB_I; i+=WI_NB)
      gscal[i + line_index] = gscal_loc[i];
}
