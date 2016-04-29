


__kernel void buff_advec(__global const float* gvelo,
			 __global float* ppos,
			 __global float* buffer_l,
			 __global float* buffer_r,
			 float dt, __constant struct AdvectionMeshInfo* mesh)
{
  int gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  int gidY = get_global_id(1); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(2); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index ; /* Current 1D problem index */

  float v,vp,p,c, hdt = 0.5 * dt;
  int i_ind, i_ind_p;

  __local float velocity_cache[V_NB_I];
  __local float buff_l_loc[V_BUFF_WIDTH];
  __local float buff_r_loc[V_BUFF_WIDTH];
  __local float* loc_ptr;

    for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB)){
      buff_l_loc[i] = buffer_l[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];
    }

    for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB)){
      buff_r_loc[i] = buffer_r[i + V_BUFF_WIDTH*(gidY + gidZ*V_NB_II)];
    }

  line_index = gidY*V_NB_I + gidZ*V_NB_I*V_NB_II;
  for(i=gidX; i<V_NB_I; i+=(WI_NB))
    {
      /* Read velocity */
      /* Fill velocity cache */
      velocity_cache[i] = gvelo[i+line_index];
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  line_index = gidY*NB_I+gidZ*NB_I*NB_II;
  for(i=gidX; i<NB_I; i+=WI_NB)
    {
      c = i * mesh->dx.x + mesh->min_position;
      v = velocity_cache[i + V_GHOSTS_NB];
      p = (c + hdt*v) * mesh->v_invdx;

      i_ind = convert_int_rtn(p);
      p = p - convert_float(i_ind);
      i_ind_p = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache + i_ind - (V_START_INDEX-V_GHOSTS_NB) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? buff_l_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1) : buff_r_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = *loc_ptr;

      loc_ptr = (i_ind_p>=(V_START_INDEX-V_GHOSTS_NB) && i_ind_p <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+i_ind_p - (V_START_INDEX-V_GHOSTS_NB) : (i_ind_p<(V_START_INDEX-V_GHOSTS_NB)) ? buff_l_loc+i_ind_p-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1) : buff_r_loc+i_ind_p-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      vp = *loc_ptr;

      v = (p*(vp-v) + v);
      p = c + dt * v;
      ppos[i+line_index] = p;
    }

}
