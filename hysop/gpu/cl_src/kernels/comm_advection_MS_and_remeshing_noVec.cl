



__kernel void buff_advec_and_remesh_l(__global const float* gvelo,
				      __global float* v_l_buff,
				      __global const float* pscal,
				      __global float* s_l_buff,
				      int used_width,
				      float dt,
				      float inv_v_dx_y, float inv_v_dx_z,
				      __constant struct AdvectionMeshInfo* mesh)
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index = gidY*NB_I+gidZ*NB_I*NB_II; /* Current 1D problem index */
  float p,v,c,s,y,w;
  float2 hY, hZ;
  int i_ind, i_indY, i_indZ;
  int ind, index;


  float velocity_cache[V_NB_I];
  float v_l_buff_loc[V_BUFF_WIDTH];
  float s_l_buff_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for (i=0; i<used_width; i++)
    s_l_buff_loc[i] = 0.0;

  barrier(CLK_LOCAL_MEM_FENCE);

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

  for (i=0; i<V_NB_I; i++){
    velocity_cache[noBC_id(i)] = hY.s1*hZ.s1 * gvelo[i + i_indY * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s1*hZ.s0 * gvelo[i + i_indY * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s1 * gvelo[i + (i_indY + 1) * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s0 * gvelo[i + (i_indY + 1) * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
  }

  for (i=0; i<V_BUFF_WIDTH; i++){
    v_l_buff_loc[i] = hY.s1*hZ.s1*v_l_buff[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    v_l_buff_loc[i] += hY.s1*hZ.s0*v_l_buff[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    v_l_buff_loc[i] += hY.s0*hZ.s1*v_l_buff[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    v_l_buff_loc[i] += hY.s0*hZ.s0*v_l_buff[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=0; i<2*BUFF_WIDTH; i++)
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
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : v_l_buff_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1);
      v = (1.0-p)*(*loc_ptr);
      i_ind = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : v_l_buff_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1);
      v += p*(*loc_ptr);
      p = c + dt * v;


      /* Read particle scalar */
      s = pscal[i + line_index];




      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index<START_INDEX) {loc_ptr = s_l_buff_loc+index-(START_INDEX-1-used_width+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store buffers
  for (i=0; i<used_width; i++)
    s_l_buff[i + gidY*used_width + gidZ*used_width*NB_II] = s_l_buff_loc[i];

}

__kernel void buff_advec_and_remesh_r(__global const float* gvelo,
				      __global float* v_r_buff,
				      __global const float* pscal,
				      __global float* s_r_buff,
				      int used_width,
				      float dt,
				      float inv_v_dx_y, float inv_v_dx_z,
				      __constant struct AdvectionMeshInfo* mesh)
{
  int gidY = get_global_id(0); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(1); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index = gidY*NB_I+gidZ*NB_I*NB_II; /* Current 1D problem index */
  float p,v,c,s,y,w;
  float2 hY, hZ;
  int i_ind, i_indY, i_indZ;
  int ind, index;


  float velocity_cache[V_NB_I];
  float v_r_buff_loc[V_BUFF_WIDTH];
  float s_r_buff_loc[BUFF_WIDTH];
  float* loc_ptr;

  // Initialize buffers
  for(i=0; i<used_width; i++)
    s_r_buff_loc[i] = 0.0;

  barrier(CLK_LOCAL_MEM_FENCE);

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

  for(i=0;i<V_NB_I; i++){
    velocity_cache[noBC_id(i)] = hY.s1*hZ.s1 * gvelo[i + i_indY * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s1*hZ.s0 * gvelo[i + i_indY * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s1 * gvelo[i + (i_indY + 1) * V_NB_I + i_indZ * V_NB_I * V_NB_II];
    velocity_cache[noBC_id(i)] += hY.s0*hZ.s0 * gvelo[i + (i_indY + 1) * V_NB_I + (i_indZ + 1) * V_NB_I * V_NB_II];
  }

  for(i=0;i<V_BUFF_WIDTH; i++){
    v_r_buff_loc[i] = hY.s1*hZ.s1*v_r_buff[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    v_r_buff_loc[i] += hY.s1*hZ.s0*v_r_buff[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    v_r_buff_loc[i] += hY.s0*hZ.s1*v_r_buff[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    v_r_buff_loc[i] += hY.s0*hZ.s0*v_r_buff[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=NB_I-2*BUFF_WIDTH; i<NB_I; i++)
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
      loc_ptr = (i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : v_r_buff_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = (1.0-p)*(*loc_ptr);
      i_ind = i_ind + 1;
      loc_ptr = (i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : v_r_buff_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v += p*(*loc_ptr);
      p = c + dt * v;


      /* Read particle scalar */
      s = pscal[i + line_index];




      ind = convert_int_rtn(p * mesh->invdx);
      y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

      index = ind - REMESH_SHIFT;

      w = REMESH(alpha)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(beta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(gamma)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(delta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
      index = index + 1;
      w = REMESH(eta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(zeta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
      index = index + 1;
      w = REMESH(theta)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(iota)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
      index = index + 1;
      w = REMESH(kappa)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);

      index = index + 1;
      w = REMESH(mu)(y);
      if (index > STOP_INDEX){ loc_ptr = s_r_buff_loc + index-(STOP_INDEX+1);
      w = w * s;
      (*loc_ptr) += w;}
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Store buffers
  for(i=0;i<used_width;i++)
    s_r_buff[i + gidY*used_width + gidZ*used_width*NB_II] = s_r_buff_loc[i];

}

__kernel void buff_advec_and_remesh(__global const float* gvelo,
				      __global float* v_l_buff,
				      __global float* v_r_buff,
				      __global const float* pscal,
				      __global float* gscal,
				      float dt,
				      float inv_v_dx_y, float inv_v_dx_z,
				      __constant struct AdvectionMeshInfo* mesh)
{
  int gidX = get_global_id(0);	/* OpenCL work-itme global index (X) */
  int gidY = get_global_id(1); /* OpenCL work-itme global index (Y) */
  int gidZ = get_global_id(2); /* OpenCL work-itme global index (Z) */
  int i;			/* Particle index in 1D problem */
  int line_index = gidY*NB_I+gidZ*NB_I*NB_II; /* Current 1D problem index */
  float p,v,c,s,y,w;
  float2 hY, hZ;
  int i_ind, i_indY, i_indZ;
  int ind, index;


  __local float velocity_cache[V_NB_I];
  __local float v_l_buff_loc[V_BUFF_WIDTH];
  __local float v_r_buff_loc[V_BUFF_WIDTH];
  __local float gscal_loc[NB_I];
  __local float* loc_ptr;

  for(i=gidX; i<NB_I; i+=WI_NB)
    {
      /* Initialize result buffer */
      gscal_loc[i] = 0.0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);



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
    v_l_buff_loc[i] = hY.s1*hZ.s1*v_l_buff[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    v_l_buff_loc[i] += hY.s1*hZ.s0*v_l_buff[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    v_l_buff_loc[i] += hY.s0*hZ.s1*v_l_buff[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    v_l_buff_loc[i] += hY.s0*hZ.s0*v_l_buff[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  for(i=gidX; i<V_BUFF_WIDTH; i+=(WI_NB)){
    v_r_buff_loc[i] = hY.s1*hZ.s1*v_r_buff[i + V_BUFF_WIDTH*(i_indY + i_indZ*V_NB_II)];
    v_r_buff_loc[i] += hY.s1*hZ.s0*v_r_buff[i + V_BUFF_WIDTH*(i_indY + (i_indZ+1)*V_NB_II)];
    v_r_buff_loc[i] += hY.s0*hZ.s1*v_r_buff[i + V_BUFF_WIDTH*(i_indY+1 + i_indZ*V_NB_II)];
    v_r_buff_loc[i] += hY.s0*hZ.s0*v_r_buff[i + V_BUFF_WIDTH*(i_indY+1 + (i_indZ+1)*V_NB_II)];
  }

  /* Synchronize work-group */
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i=gidX*PART_NB_PER_WI; i<(gidX + 1)*PART_NB_PER_WI; i+=1)
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
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? v_l_buff_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1)  : v_r_buff_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v = (1.0-p)*(*loc_ptr);
      i_ind = i_ind + 1;
      loc_ptr = (i_ind>=(V_START_INDEX-V_GHOSTS_NB) && i_ind <= (V_STOP_INDEX+V_GHOSTS_NB)) ? velocity_cache+noBC_id(i_ind - (V_START_INDEX-V_GHOSTS_NB)) : (i_ind<(V_START_INDEX-V_GHOSTS_NB)) ? v_l_buff_loc+i_ind-(V_START_INDEX-V_GHOSTS_NB-1-V_BUFF_WIDTH+1)  : v_r_buff_loc+i_ind-(V_STOP_INDEX+V_GHOSTS_NB+1) ;
      v += p*(*loc_ptr);
      p = c + dt * v;


      /* Read particle scalar */
      s = pscal[i + line_index];




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
