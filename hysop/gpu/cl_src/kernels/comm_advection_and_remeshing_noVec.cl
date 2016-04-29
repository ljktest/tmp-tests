


__kernel void buff_advec_and_remesh_l(__global const float* gvelo,
				      __global float* v_buffer_l,
				      __global const float* pscal,
				      __global float* s_buffer_l,
				      int used_width,
				      float dt, __constant struct AdvectionMeshInfo* mesh)
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index ; /* Current 1D problem index */

  float v,vp,p,c,s,y,w, hdt = 0.5 * dt;
  int i_ind, i_ind_p, ind, index;

  float velocity_cache[V_NB_I];
  float v_buff_l_loc[V_BUFF_WIDTH];
  float s_buff_l_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for (i=0;i<used_width;i++)
    s_buff_l_loc[i] = 0.0;

  for(i=0; i<V_BUFF_WIDTH; i++)
    v_buff_l_loc[i] = v_buffer_l[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];

  line_index = gidY*V_NB_I + gidZ*V_NB_I*V_NB_II;
  /* Read velocity */
  /* Fill velocity cache */
  for(i=0;i<V_NB_I;i++)
      velocity_cache[i] = gvelo[i+line_index];

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  line_index = gidY*NB_I+gidZ*NB_I*NB_II;
  for(i=0; i<2*BUFF_WIDTH; i++)
    {
      /* Read particle scalar */
      s = pscal[i + line_index];

      c = i * mesh->dx.x + mesh->min_position;
      v = velocity_cache[i + V_GHOSTS_NB];
      p = (c + hdt*v) * mesh->v_invdx;

      i_ind = convert_int_rtn(p);
      p = p - convert_float(i_ind);
      i_ind_p = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB)) ? velocity_cache + i_ind - (V_START_INDEX-V_GHOSTS_NB) : v_buff_l_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1);
      v = *loc_ptr;

      loc_ptr = (i_ind_p>=(V_START_INDEX-V_GHOSTS_NB)) ? velocity_cache+i_ind_p - (V_START_INDEX-V_GHOSTS_NB) : v_buff_l_loc+i_ind_p-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1);
      vp = *loc_ptr;

      v = (p*(vp-v) + v);
      p = c + dt * v;



      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index<START_INDEX){ loc_ptr = s_buff_l_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store buffers
  for(i=0;i<used_width;i++)
    s_buffer_l[i + gidY*used_width + gidZ*used_width*NB_II] = s_buff_l_loc[i];
}








__kernel void buff_advec_and_remesh_r(__global const float* gvelo,
				      __global float* v_buffer_r,
				      __global const float* pscal,
				      __global float* s_buffer_r,
				      int used_width,
				      float dt, __constant struct AdvectionMeshInfo* mesh)
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index ; /* Current 1D problem index */

  float v,vp,p,c,s,y,w, hdt = 0.5 * dt;
  int i_ind, i_ind_p, ind, index;

  float velocity_cache[V_NB_I];
  float v_buff_r_loc[V_BUFF_WIDTH];
  float s_buff_r_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for(i=0;i<used_width;i++)
    s_buff_r_loc[i] = 0.0;

  for(i=0;i<V_BUFF_WIDTH;i++)
    v_buff_r_loc[i] = v_buffer_r[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];

  line_index = gidY*V_NB_I + gidZ*V_NB_I*V_NB_II;
  /* Read velocity */
  /* Fill velocity cache */
  for(i=0;i<V_NB_I; i++)
      velocity_cache[i] = gvelo[i+line_index];

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  line_index = gidY*NB_I+gidZ*NB_I*NB_II;
  for(i=NB_I-2*BUFF_WIDTH; i<NB_I; i++)
    {
      /* Read particle scalar */
      s = pscal[i + line_index];

      c = i * mesh->dx.x + mesh->min_position;
      v = velocity_cache[i + V_GHOSTS_NB];
      p = (c + hdt*v) * mesh->v_invdx;

      i_ind = convert_int_rtn(p);
      p = p - convert_float(i_ind);
      i_ind_p = i_ind + 1;
      loc_ptr = (i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache + i_ind - (V_START_INDEX-V_GHOSTS_NB) : v_buff_r_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = *loc_ptr;

      loc_ptr = (i_ind_p <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+i_ind_p - (V_START_INDEX-V_GHOSTS_NB) : v_buff_r_loc+i_ind_p-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      vp = *loc_ptr;

      v = (p*(vp-v) + v);
      p = c + dt * v;



      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index > STOP_INDEX){ loc_ptr = s_buff_r_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=0;i<used_width;i++)
    s_buffer_r[i + gidY*used_width + gidZ*used_width*NB_II] = s_buff_r_loc[i];

}


__kernel void buff_advec_and_remesh(__global const float* gvelo,
				    __global float* v_buffer_l,
				    __global float* v_buffer_r,
				    __global const float* pscal,
				    __global float* gscal,
				    float dt, __constant struct AdvectionMeshInfo* mesh)
{
  int gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  int gidY = get_global_id(1); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(2); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index ; /* Current 1D problem index */

  float v,vp,p,c,s,y,w, hdt = 0.5 * dt;
  int i_ind, i_ind_p, ind, index;

  __local float velocity_cache[V_NB_I];
  __local float v_buff_l_loc[V_BUFF_WIDTH];
  __local float v_buff_r_loc[V_BUFF_WIDTH];
  __local float gscal_loc[NB_I];
  __local float* loc_ptr;


  for(i=gidX; i<NB_I; i+=WI_NB)
    /* Initialize result buffer */
    gscal_loc[i] = 0.0;

  for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB))
    v_buff_l_loc[i] = v_buffer_l[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];

  for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB))
    v_buff_r_loc[i] = v_buffer_r[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];

  line_index = gidY*V_NB_I + gidZ*V_NB_I*V_NB_II;
  /* Read velocity */
  /* Fill velocity cache */
  for(i=gidX; i<V_NB_I; i+=(WI_NB))
      velocity_cache[i] = gvelo[i+line_index];

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  line_index = gidY*NB_I+gidZ*NB_I*NB_II;
  for(i=gidX*PART_NB_PER_WI; i<(gidX + 1)*PART_NB_PER_WI; i+=1)
    {
      /* Read particle scalar */
      s = pscal[i + line_index];

      c = i * mesh->dx.x + mesh->min_position;
      v = velocity_cache[i + V_GHOSTS_NB];
      p = (c + hdt*v) * mesh->v_invdx;

      i_ind = convert_int_rtn(p);
      p = p - convert_float(i_ind);
      i_ind_p = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache + i_ind - (V_START_INDEX-V_GHOSTS_NB) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? v_buff_l_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1) : v_buff_r_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = *loc_ptr;

      loc_ptr = (i_ind_p>=(V_START_INDEX-V_GHOSTS_NB) && i_ind_p <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+i_ind_p - (V_START_INDEX-V_GHOSTS_NB) : (i_ind_p<(V_START_INDEX-V_GHOSTS_NB)) ? v_buff_l_loc+i_ind_p-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1) : v_buff_r_loc+i_ind_p-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      vp = *loc_ptr;

      v = (p*(vp-v) + v);
      p = c + dt * v;



      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index>=START_INDEX && index <= STOP_INDEX){ loc_ptr = gscal_loc +index-START_INDEX;
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=gidX; i<NB_I; i+=WI_NB)
    {
      /* Store result */
      gscal[i + line_index] = gscal_loc[i];
    }
}
