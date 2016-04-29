



__kernel void buff_advec(__global const float* gvelo,
			 __global float* ppos,
			 __global float* buffer_l,
			 __global float* buffer_r,
			 float dt,
			 float inv_v_dx_y, float inv_v_dx_z,
			 __constant struct AdvectionMeshInfo* mesh)
{
  int gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  int gidY = get_global_id(1); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(2); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index = gidY*NB_I+gidZ*NB_I*NB_II; /* Current 1D problem index */
  float p,v,c;
  float2 hY, hZ;
  int i_ind, i_indY, i_indZ;


  __local float velocity_cache[V_NB_I];
  __local float buff_l_loc[V_BUFF_WIDTH];
  __local float buff_r_loc[V_BUFF_WIDTH];
  __local float* loc_ptr;


  hY.s0 = (gidY * mesh->dx.y) * inv_v_dx_y;
  hZ.s0 = (gidZ * mesh->dx.z) * inv_v_dx_z;
  i_indY = convert_int_rtn(hY.s0);
  i_indZ = convert_int_rtn(hZ.s0);
  hY.s0 = hY.s0 - convert_float(i_indY);
  hZ.s0 = hZ.s0 - convert_float(i_indZ);
  hY.s1 = (1.0-hY.s0);
  hZ.s1 = (1.0-hZ.s0);

  i_indY = i_indY + V_GHOSTS_NB;
  i_indZ = i_indZ + V_GHOSTS_NB;

  for(i=gidX; i<V_NB_I; i+=(WI_NB)){
    velocity_cache[noBC_id(i)] = hY.s1*hZ.s1 * gvelo[i + i_indY * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s1*hZ.s0 * gvelo[i + i_indY * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s1 * gvelo[i + (i_indY + 1) * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s0 * gvelo[i + (i_indY + 1) * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
  }

  for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB)){
    buff_l_loc[i] = hY.s1*hZ.s1*buffer_l[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    buff_l_loc[i] += hY.s1*hZ.s0*buffer_l[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    buff_l_loc[i] += hY.s0*hZ.s1*buffer_l[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    buff_l_loc[i] += hY.s0*hZ.s0*buffer_l[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB)){
    buff_r_loc[i] = hY.s1*hZ.s1*buffer_r[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    buff_r_loc[i] += hY.s1*hZ.s0*buffer_r[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    buff_r_loc[i] += hY.s0*hZ.s1*buffer_r[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    buff_r_loc[i] += hY.s0*hZ.s0*buffer_r[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=gidX; i<NB_I; i+=WI_NB)
    {
      c = i * mesh->dx.x + mesh->min_position;
      // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
      p = c * mesh->v_invdx;
      i_ind = convert_int_rtn(p);
      p = p - convert_float(i_ind);
      i_ind = i_ind - (V_START_INDEX-V_GHOSTS_NB) - MS_INTERPOL_SHIFT;
      v = mix(velocity_cache[noBC_id(i_ind)],
	      velocity_cache[noBC_id(i_ind+1)],p);
      p = (c + 0.5*dt*v) * mesh->v_invdx;

      i_ind = convert_int_rtn(p) - MS_INTERPOL_SHIFT;
      p = p - convert_float(i_ind);
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? buff_l_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1)  : buff_r_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = (1.0-p)*(*loc_ptr);
      i_ind = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? buff_l_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1)  : buff_r_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v += p*(*loc_ptr);
      ppos[i+line_index] = c + dt * v;
    }

}
