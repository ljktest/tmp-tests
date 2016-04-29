/** Computes the right hand side of the baroclinic term.
 * The pressure gradient is given in input at a coarse scale.
 * The density is given at a fine scale.
 * Result is computed ans returned at fine scale.
 */
#define C_TILE_IDX(x,y) x+GHOSTS_C_X+(y+GHOSTS_C_Y)*C_TILE_WIDTH
#if FD_ORDER == FD_C_2
#define GRAD_GH 1
#endif
#if FD_ORDER == FD_C_4
#define GRAD_GH 2
#endif

float compute_density(float x);
float compute_density(float x){
  return __USER_DENSITY_FUNCTION_FROM_GIVEN_INPUT__;
}

float interpolate(__local float* loc_gradp_zm, __local float* loc_gradp_zp,
		  float *h, int lidx, int lidy,
		  int cellx, int celly, int cellz);

float interpolate(__local float* loc_zm,
		  __local float* loc_zp,
		  float *h,
		  int lidx,
		  int lidy,
		  int cellx,
		  int celly,
		  int cellz) {
  float res = 0.0;
  res += (1.0 - h[cellz]) * (1.0 - h[cellx]) * (1.0 - h[celly]) * loc_zm[C_TILE_IDX(lidx,lidy)];
  res += (1.0 - h[cellz]) * (h[cellx]) * (1.0 - h[celly]) * loc_zm[C_TILE_IDX(lidx+1,lidy)];
  res += (1.0 - h[cellz]) * (1.0 - h[cellx]) * (h[celly]) * loc_zm[C_TILE_IDX(lidx,lidy+1)];
  res += (1.0 - h[cellz]) * (h[cellx]) * (h[celly]) * loc_zm[C_TILE_IDX(lidx+1,lidy+1)];
  res += (h[cellz]) * (1.0 - h[cellx]) * (1.0 - h[celly]) * loc_zp[C_TILE_IDX(lidx,lidy)];
  res += (h[cellz]) * (h[cellx]) * (1.0 - h[celly]) * loc_zp[C_TILE_IDX(lidx+1,lidy)];
  res += (h[cellz]) * (1.0 - h[cellx]) * (h[celly]) * loc_zp[C_TILE_IDX(lidx,lidy+1)];
  res += (h[cellz]) * (h[cellx]) * (h[celly]) * loc_zp[C_TILE_IDX(lidx+1,lidy+1)];
  return res;
}

void fill_loc_rho_cache(__local float *loc_rho,
			__global const float* rho,
#if CUT_DIR_Y == 1
			__global const float* rho_ghostsY,
#endif
			int lidx,
			int lidy,
			int gidx,
			int gidy,
			int idz);
void fill_loc_rho_cache(__local float *loc_rho,
			__global const float* rho,
#if CUT_DIR_Y == 1
			__global const float* rho_ghostsY,
#endif
			int lidx,
			int lidy,
			int gidx,
			int gidy,
			int idz) {
  int celly, cellx;
  if (gidx > 0 && gidx < ((int)get_num_groups(0))-1 && gidy > 0 && gidy < ((int)get_num_groups(1))-1) {
    for (celly=lidy; celly<F_TILE_SIZE+2*GRAD_GH; celly+=get_local_size(1)) {
      for (cellx=lidx; cellx<F_TILE_SIZE+2*GRAD_GH; cellx+=get_local_size(0)) {
	loc_rho[cellx + (celly)*(F_TILE_SIZE+2*GRAD_GH)] =
	  compute_density(rho[cellx-GRAD_GH + gidx*F_TILE_SIZE +
			      (celly-GRAD_GH + gidy*F_TILE_SIZE)*NB_F_X +
			      idz*NB_F_X*NB_F_Y]);
      }
    }
  } else {
    for (celly=lidy; celly<F_TILE_SIZE+2*GRAD_GH; celly+=get_local_size(1)) {
      for (cellx=lidx; cellx<F_TILE_SIZE+2*GRAD_GH; cellx+=get_local_size(0)) {
#if CUT_DIR_Y == 1
	if (celly-GRAD_GH + gidy*F_TILE_SIZE >= NB_F_Y)
	  loc_rho[cellx + (celly)*(F_TILE_SIZE+2*GRAD_GH)] =
	    compute_density(rho_ghostsY[(cellx-GRAD_GH + gidx*F_TILE_SIZE+NB_F_X)%NB_F_X +
					(celly-GRAD_GH + gidy*F_TILE_SIZE - NB_F_Y)*NB_F_X +
					idz*NB_F_X*2*GRAD_GH]);
	else if (celly-GRAD_GH + gidy*F_TILE_SIZE < 0)
	  loc_rho[cellx + (celly)*(F_TILE_SIZE+2*GRAD_GH)] =
	    compute_density(rho_ghostsY[(cellx-GRAD_GH + gidx*F_TILE_SIZE+NB_F_X)%NB_F_X +
					(2*GRAD_GH + (celly-GRAD_GH + gidy*F_TILE_SIZE))*NB_F_X +
					idz*NB_F_X*2*GRAD_GH]);
	else
	  loc_rho[cellx + (celly)*(F_TILE_SIZE+2*GRAD_GH)] =
	    compute_density(rho[(cellx-GRAD_GH + gidx*F_TILE_SIZE+NB_F_X)%NB_F_X +
				(celly-GRAD_GH + gidy*F_TILE_SIZE)*NB_F_X +
				idz*NB_F_X*NB_F_Y]);
#else
	loc_rho[cellx + (celly)*(F_TILE_SIZE+2*GRAD_GH)] =
	  compute_density(rho[(cellx-GRAD_GH + gidx*F_TILE_SIZE+NB_F_X)%NB_F_X +
			      ((celly-GRAD_GH + gidy*F_TILE_SIZE+NB_F_Y)%NB_F_Y)*NB_F_X +
			      idz*NB_F_X*NB_F_Y]);
#endif
      }
    }
  }
}

__kernel void baroclinic_rhs(__global float* rhs_x,
			     __global float* rhs_y,
			     __global float* rhs_z,
			     __global const float* rho,
#if CUT_DIR_Y == 1
			     __global const float* rho_ghostsY,
#endif
#if CUT_DIR_Z == 1
			     __global const float* rho_ghostsZ,
#endif
			     __global const float* gradp,
			     float4 dx_coarse,
			     float4 dx_fine)
{
  /* Space index refers to the coarse grid comute points */
  int lidx = get_local_id(0);
  int lidy = get_local_id(1);
  int gidx = get_group_id(0);
  int gidy = get_group_id(1);
  int rhs_idx = lidx*N_PER_CELL + gidx*F_TILE_SIZE + (lidy*N_PER_CELL + gidy*F_TILE_SIZE)*NB_F_X;
  float h[N_PER_CELL];
  int i, cellx, celly, cellz;
  int idz, c_idz;
  float p_gradp, gradrho_x, gradrho_y, gradrho_z;
  float rho_zm[N_PER_CELL][N_PER_CELL];
#if FD_ORDER == FD_C_4
  float rho_zmm[N_PER_CELL][N_PER_CELL];
  float rho_zp[N_PER_CELL][N_PER_CELL];
  float rho_zpp;
#endif

  __local float loc_rho[(F_TILE_SIZE+2*GRAD_GH)*(F_TILE_SIZE+2*GRAD_GH)];
  __local float loc_gradp_zm[C_TILE_WIDTH*C_TILE_HEIGHT];
  __local float loc_gradp_zp[C_TILE_WIDTH*C_TILE_HEIGHT];


  // Compute distances from fine grid points to coarse left point cell.
  for (i=0; i<N_PER_CELL; i++)
    h[i] = i * 1.0 / (1.0 * N_PER_CELL);

  idz = 0; 			/* Fine grid Z indice */
  c_idz=GHOSTS_C_Z;
  // Fill gradp z cache for first iteration
  for (celly=lidy; celly<C_TILE_HEIGHT; celly+=get_local_size(1)) {
    for (cellx=lidx; cellx<C_TILE_WIDTH; cellx+=get_local_size(0)) {
      loc_gradp_zm[cellx + celly*(C_TILE_WIDTH)] =
	gradp[cellx + gidx*C_TILE_SIZE + (celly + gidy*C_TILE_SIZE)*NB_C_X + c_idz*NB_C_X*NB_C_Y];
    }
  }

  for (celly=0; celly<N_PER_CELL; celly++) {
    for (cellx=0; cellx<N_PER_CELL; cellx++) {
#if FD_ORDER == FD_C_4
#if CUT_DIR_Z == 1
      rho_zm[cellx][celly] = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X + 3*NB_F_X*NB_F_Y]);
      rho_zmm[cellx][celly] = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X + 2*NB_F_X*NB_F_Y]);
      rho_zp[cellx][celly] = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + NB_F_X*NB_F_Y]);
#else
      rho_zm[cellx][celly] = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + (NB_F_Z-1)*NB_F_X*NB_F_Y]);
      rho_zmm[cellx][celly] = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + (NB_F_Z-2)*NB_F_X*NB_F_Y]);
      rho_zp[cellx][celly] = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + NB_F_X*NB_F_Y]);
#endif
#else
#if CUT_DIR_Z == 1
      rho_zm[cellx][celly] = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X + NB_F_X*NB_F_Y]);
#else
      rho_zm[cellx][celly] = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + (NB_F_Z-1)*NB_F_X*NB_F_Y]));
#endif
#endif
    }
  }

  for (c_idz=GHOSTS_C_Z; c_idz<NB_C_Z-GHOSTS_C_Z; c_idz++) {

    if((c_idz-GHOSTS_C_Z)%2 == 0)
      for (celly=lidy; celly<C_TILE_HEIGHT; celly+=get_local_size(1)) {
	for (cellx=lidx; cellx<C_TILE_WIDTH; cellx+=get_local_size(0)) {
	  loc_gradp_zp[cellx + celly*(C_TILE_WIDTH)] =
	    gradp[cellx + gidx*C_TILE_SIZE + (celly + gidy*C_TILE_SIZE)*NB_C_X + (c_idz+1)*NB_C_X*NB_C_Y];
	}
      }
    else
      for (celly=lidy; celly<C_TILE_HEIGHT; celly+=get_local_size(1)) {
	for (cellx=lidx; cellx<C_TILE_WIDTH; cellx+=get_local_size(0)) {
	  loc_gradp_zm[cellx + celly*(C_TILE_WIDTH)] =
	    gradp[cellx + gidx*C_TILE_SIZE + (celly + gidy*C_TILE_SIZE)*NB_C_X + (c_idz+1)*NB_C_X*NB_C_Y];
	}
      }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (cellz=0; cellz<N_PER_CELL; cellz++) {
      //fill rho cache
#if CUT_DIR_Y == 1
      fill_loc_rho_cache(loc_rho, rho, rho_ghostsY, lidx, lidy, gidx, gidy, idz);
#else
      fill_loc_rho_cache(loc_rho, rho, lidx, lidy, gidx, gidy, idz);
#endif

      barrier(CLK_LOCAL_MEM_FENCE);

      for (celly=0; celly<N_PER_CELL; celly++) {
	for (cellx=0; cellx<N_PER_CELL; cellx++) {
	  if((c_idz-GHOSTS_C_Z)%2 == 0)
	    p_gradp = interpolate(loc_gradp_zm, loc_gradp_zp, h, lidx, lidy, cellx, celly, cellz);
	  else
	    p_gradp = interpolate(loc_gradp_zp, loc_gradp_zm, h, lidx, lidy, cellx, celly, cellz);

	  ///// TEMP WRITE GRADP TO RHS
	  //rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = p_gradp;
	  ///// END TEMP WRITE GRADP TO RHS

#if FD_ORDER == FD_C_2
	  gradrho_x = loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx+1 +
			      (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx-1 +
			       (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x /= (2.0*dx_fine.x);

	  gradrho_y = loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			      (GRAD_GH+lidy*N_PER_CELL+celly+1)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			       (GRAD_GH+lidy*N_PER_CELL+celly-1)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y /= (2.0*dx_fine.y);

#if CUT_DIR_Z == 1
	  if (idz==NB_F_Z-1)
	    gradrho_z = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X]);
	  else
	    gradrho_z = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + ((idz+1)%NB_F_Z)*NB_F_X*NB_F_Y]);
#else
 	  gradrho_z = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + ((idz+1)%NB_F_Z)*NB_F_X*NB_F_Y]);
#endif
	  gradrho_z -= rho_zm[cellx][celly];
	  gradrho_z /= (2.0*dx_fine.z);

#endif
#if FD_ORDER == FD_C_4
	  gradrho_x = loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx+1 +
			      (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx-1 +
			      (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x *= 8.0;
	  gradrho_x += loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx-2 +
			      (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx+2 +
			      (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_x /= (12.0*dx_fine.x);

	  gradrho_y = loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			      (GRAD_GH+lidy*N_PER_CELL+celly+1)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			      (GRAD_GH+lidy*N_PER_CELL+celly-1)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y *= 8.0;
	  gradrho_y += loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			      (GRAD_GH+lidy*N_PER_CELL+celly-2)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y -= loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
			      (GRAD_GH+lidy*N_PER_CELL+celly+2)*(F_TILE_SIZE+2*GRAD_GH)];
	  gradrho_y /= (12.0*dx_fine.y);

#if CUT_DIR_Z == 1
	  if (idz==NB_F_Z-1)
	    rho_zpp = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X + NB_F_X*NB_F_Y]);
	  else if (idz==NB_F_Z-2)
	    rho_zpp = compute_density(rho_ghostsZ[rhs_idx + cellx + celly*NB_F_X]);
	  else
	    rho_zpp = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + ((idz+2)%NB_F_Z)*NB_F_X*NB_F_Y]);
#else
	  rho_zpp = compute_density(rho[rhs_idx + cellx + celly*NB_F_X + ((idz+2)%NB_F_Z)*NB_F_X*NB_F_Y]);
#endif
	  gradrho_z = rho_zp[cellx][celly];
	  gradrho_z -= rho_zm[cellx][celly];
	  gradrho_z *= 8.0;
	  gradrho_z += rho_zmm[cellx][celly];
	  gradrho_z -= rho_zpp;
	  gradrho_z /= (12.0*dx_fine.z);
#endif

	  ///// TEMP WRITE GRADrho_X TO RHS
	  //rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = gradrho_x;
	  ///// END TEMP WRITE GRADrho TO RHS
	  ///// TEMP WRITE GRADrho_Y TO RHS
	  //rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = gradrho_y;
	  ///// END TEMP WRITE GRADrho TO RHS
	  ///// TEMP WRITE GRADrho_Z TO RHS
	  //rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = gradrho_z;
	  ///// END TEMP WRITE GRADrho TO RHS

	  // Using gradp X component as gradp and assuming this kernel run first to initialise output
#if GRADP_COMP == 0
	  rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = 0.0;
	  rhs_y[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = -gradrho_z*p_gradp;
	  rhs_z[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] = gradrho_y*p_gradp;
#endif
	  // Using gradp Y component as gradp
#if GRADP_COMP == 1
	  rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] += gradrho_z*p_gradp;
	  rhs_z[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] -= gradrho_x*p_gradp;
#endif
	  // Using gradp Z component as gradp
#if GRADP_COMP == 2
	  rhs_x[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] -= gradrho_y*p_gradp;
	  rhs_y[rhs_idx + cellx + celly*NB_F_X + idz*NB_F_X*NB_F_Y] += gradrho_x*p_gradp;
#endif


	  // For next iteration we swap values in cache.
#if FD_ORDER == FD_C_4
	  rho_zp[cellx][celly] = rho_zpp;
	  rho_zmm[cellx][celly] = rho_zm[cellx][celly];
#endif
	  rho_zm[cellx][celly] = loc_rho[GRAD_GH+lidx*N_PER_CELL+cellx +
					 (GRAD_GH+lidy*N_PER_CELL+celly)*(F_TILE_SIZE+2*GRAD_GH)];
	}
      }
      idz++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
