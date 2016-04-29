__kernel void initialize_output(__global float* scal_out) {
  scal_out[get_global_id(0) + get_global_id(1)*NB_OUT_X + get_global_id(2)*NB_OUT_X*NB_OUT_Y] = 0.0;
}

__kernel void coarse_to_fine_filter(__global const float* scal_in,
				    __global float* scal_out,
				    float scale_factor,
				    float4 dx_in, float4 dx_out, float4 origin,
				    int offset_y, int offset_z) {
  // Work-group is computed from coarse grid (without ghosts)
  // globalsize(1) = (NB_OUT_Y - 2*GHOSTS_OUT_Y) / PTS_PER_CELL_Y
  // globalsize(2) = (NB_OUT_Z - 2*GHOSTS_OUT_Z) / PTS_PER_CELL_X
  // Resolutions are linked by: (NB_OUT - 2*GHOSTS_OUT) * PTS_PER_CELL = NB_IN
  // A work-group is in charge of a subdomain corresponding to:
  //   - [NB_OUT_X, L_STENCIL, L_STENCIL] for the coarse grid
  //   - [NB_IN_X, PTS_PER_CELL_Y, PTS_PER_CELL] for the fine grid
  // Data in the fine grid are read only once for the whole computation.
  // Because of the stencil, these data are spread over multiple coarse grid cells -> we need a global memory synchronization.
  // The global synchronization is obtained by several kernel launch with an offset
  unsigned int lid = get_local_id(0);
  unsigned int gid_y = get_global_id(1);
  unsigned int gid_z = get_global_id(2);
  unsigned int iy_c = gid_y*L_STENCIL+offset_y;
  unsigned int iz_c = gid_z*L_STENCIL+offset_z;
  unsigned int iy_f = iy_c*PTS_PER_CELL_Y;
  unsigned int iz_f = iz_c*PTS_PER_CELL_Z;
  unsigned int i, j, k, b_id, pt_x, pt_y, pt_z;
  float4 coord_in;
  float4 coord_out;
  float4 d;
#if FORMULA==L2_1
  float4 wx, wy, wz;
#endif
  __local float line[WG*PTS_PER_CELL_X];
  __local float result[NB_OUT_X][L_STENCIL][L_STENCIL];
  __private float p_res[L_STENCIL][L_STENCIL][L_STENCIL];

  // Fill local arrays
  // Output data
  for (k=0;k<L_STENCIL;k++)
    for (j=0;j<L_STENCIL;j++)
      for (i=lid;i<NB_OUT_X;i+=WG)
  	result[i][j][k] = scal_out[i + (GHOSTS_OUT_Y+iy_c-SHIFT_STENCIL+j)*NB_OUT_X +
				   (GHOSTS_OUT_Z+iz_c-SHIFT_STENCIL+k)*NB_OUT_X*NB_OUT_Y];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (b_id=0;b_id<NB_IN_X/(WG*PTS_PER_CELL_X);b_id++)
    {
      // Compute a bloc of: (b_id is the bloc number in X direction)
      //   - [WG*PTS_PER_CELL_X, PTS_PER_CELL_Y, PTS_PER_CELL_Z] points in fine grid
      //   - [WG, L_STENCIL, L_STENCIL] points in coarse grid
      // Each work-item is computing a coarse cell (looping in 3D over PTS_PER_CELL thanks to pt_x, pt_y and pt_z indices)
      // global fine grid data are cached line by line in the X direction
      coord_out = ((float4)(b_id*WG+lid, iy_c, iz_c, 0.0)) * dx_out;
      // Initialize the register corresponding to the current cell
      for (pt_z=0;pt_z<L_STENCIL;pt_z++)
	for (pt_y=0;pt_y<L_STENCIL;pt_y++)
	  for (pt_x=0;pt_x<L_STENCIL;pt_x++)
	    p_res[pt_x][pt_y][pt_z] = 0.0;

      // Loop over PTS_PER_CELL_Z: fine grid points in the curent cell
      for (pt_z=0;pt_z<PTS_PER_CELL_Z;pt_z++)
	{
	  // Loop over PTS_PER_CELL_Y: fine grid points in the curent cell
	  for (pt_y=0;pt_y<PTS_PER_CELL_Y;pt_y++)
	    {
	      // Input cache
	      for (i=lid;i<WG*PTS_PER_CELL_X;i+=WG)
		line[i] = scal_in[b_id*(WG*PTS_PER_CELL_X) + i + (iy_f+pt_y)*NB_IN_X + (iz_f+pt_z)*NB_IN_X*NB_IN_Y];
	      barrier(CLK_LOCAL_MEM_FENCE);

	      // Loop over PTS_PER_CELL_X: fine grid points in the curent cell
	      for (pt_x=0;pt_x<PTS_PER_CELL_X;pt_x++)
		{
		  coord_in = ((float4)(b_id*(WG*PTS_PER_CELL_X) + lid*PTS_PER_CELL_X + pt_x, iy_f+pt_y, iz_f+pt_z, 0.0)) * dx_in;
		  d = (coord_in  - coord_out) / dx_out;
		  #if FORMULA==LINEAR
		  p_res[0][0][0] += scale_factor * (1.0 - d.x) * (1.0 - d.y) * (1.0 - d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][0][1] += scale_factor * (1.0 - d.x) * (1.0 - d.y) * (d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][0] += scale_factor * (1.0 - d.x) * (d.y) * (1.0 - d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][1] += scale_factor * (1.0 - d.x) * (d.y) * (d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][0][0] += scale_factor * (d.x) * (1.0 - d.y) * (1.0 - d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][0][1] += scale_factor * (d.x) * (1.0 - d.y) * (d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][0] += scale_factor * (d.x) * (d.y) * (1.0 - d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][1] += scale_factor * (d.x) * (d.y) * (d.z) * line[lid*PTS_PER_CELL_X+pt_x];
		  #elif FORMULA==L2_1
		  wx = (float4)(alpha_l2_1(d.x), beta_l2_1(d.x), gamma_l2_1(d.x), delta_l2_1(d.x));
		  wy = (float4)(alpha_l2_1(d.y), beta_l2_1(d.y), gamma_l2_1(d.y), delta_l2_1(d.y));
		  wz = (float4)(alpha_l2_1(d.z), beta_l2_1(d.z), gamma_l2_1(d.z), delta_l2_1(d.z));
		  p_res[0][0][0] += scale_factor * wx.x * wy.x * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][0][1] += scale_factor * wx.x * wy.x * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][0][2] += scale_factor * wx.x * wy.x * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][0][3] += scale_factor * wx.x * wy.x * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][0] += scale_factor * wx.x * wy.y * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][1] += scale_factor * wx.x * wy.y * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][2] += scale_factor * wx.x * wy.y * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][1][3] += scale_factor * wx.x * wy.y * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][2][0] += scale_factor * wx.x * wy.z * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][2][1] += scale_factor * wx.x * wy.z * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][2][2] += scale_factor * wx.x * wy.z * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][2][3] += scale_factor * wx.x * wy.z * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][3][0] += scale_factor * wx.x * wy.w * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][3][1] += scale_factor * wx.x * wy.w * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][3][2] += scale_factor * wx.x * wy.w * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[0][3][3] += scale_factor * wx.x * wy.w * wz.w * line[lid*PTS_PER_CELL_X+pt_x];

		  p_res[1][0][0] += scale_factor * wx.y * wy.x * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][0][1] += scale_factor * wx.y * wy.x * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][0][2] += scale_factor * wx.y * wy.x * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][0][3] += scale_factor * wx.y * wy.x * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][0] += scale_factor * wx.y * wy.y * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][1] += scale_factor * wx.y * wy.y * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][2] += scale_factor * wx.y * wy.y * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][1][3] += scale_factor * wx.y * wy.y * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][2][0] += scale_factor * wx.y * wy.z * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][2][1] += scale_factor * wx.y * wy.z * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][2][2] += scale_factor * wx.y * wy.z * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][2][3] += scale_factor * wx.y * wy.z * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][3][0] += scale_factor * wx.y * wy.w * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][3][1] += scale_factor * wx.y * wy.w * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][3][2] += scale_factor * wx.y * wy.w * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[1][3][3] += scale_factor * wx.y * wy.w * wz.w * line[lid*PTS_PER_CELL_X+pt_x];

		  p_res[2][0][0] += scale_factor * wx.z * wy.x * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][0][1] += scale_factor * wx.z * wy.x * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][0][2] += scale_factor * wx.z * wy.x * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][0][3] += scale_factor * wx.z * wy.x * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][1][0] += scale_factor * wx.z * wy.y * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][1][1] += scale_factor * wx.z * wy.y * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][1][2] += scale_factor * wx.z * wy.y * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][1][3] += scale_factor * wx.z * wy.y * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][2][0] += scale_factor * wx.z * wy.z * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][2][1] += scale_factor * wx.z * wy.z * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][2][2] += scale_factor * wx.z * wy.z * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][2][3] += scale_factor * wx.z * wy.z * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][3][0] += scale_factor * wx.z * wy.w * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][3][1] += scale_factor * wx.z * wy.w * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][3][2] += scale_factor * wx.z * wy.w * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[2][3][3] += scale_factor * wx.z * wy.w * wz.w * line[lid*PTS_PER_CELL_X+pt_x];

		  p_res[3][0][0] += scale_factor * wx.w * wy.x * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][0][1] += scale_factor * wx.w * wy.x * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][0][2] += scale_factor * wx.w * wy.x * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][0][3] += scale_factor * wx.w * wy.x * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][1][0] += scale_factor * wx.w * wy.y * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][1][1] += scale_factor * wx.w * wy.y * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][1][2] += scale_factor * wx.w * wy.y * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][1][3] += scale_factor * wx.w * wy.y * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][2][0] += scale_factor * wx.w * wy.z * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][2][1] += scale_factor * wx.w * wy.z * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][2][2] += scale_factor * wx.w * wy.z * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][2][3] += scale_factor * wx.w * wy.z * wz.w * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][3][0] += scale_factor * wx.w * wy.w * wz.x * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][3][1] += scale_factor * wx.w * wy.w * wz.y * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][3][2] += scale_factor * wx.w * wy.w * wz.z * line[lid*PTS_PER_CELL_X+pt_x];
		  p_res[3][3][3] += scale_factor * wx.w * wy.w * wz.w * line[lid*PTS_PER_CELL_X+pt_x];

		  #endif
		}
	    }
	}
      // Store the registers results in local memory
      for (pt_z=0;pt_z<L_STENCIL;pt_z++)
	for (pt_y=0;pt_y<L_STENCIL;pt_y++)
	  for (pt_x=0;pt_x<L_STENCIL;pt_x++) {
	    result[GHOSTS_OUT_X+b_id*WG+lid-SHIFT_STENCIL+pt_x][pt_y][pt_z] += p_res[pt_x][pt_y][pt_z];
	    barrier(CLK_LOCAL_MEM_FENCE);
	  }
    }

  // Write result in output array
  for (k=0;k<L_STENCIL;k++)
    for (j=0;j<L_STENCIL;j++)
      for (i=lid;i<NB_OUT_X;i+=WG)
	scal_out[i + (GHOSTS_OUT_Y+iy_c-SHIFT_STENCIL+j)*NB_OUT_X +
		 (GHOSTS_OUT_Z+iz_c-SHIFT_STENCIL+k)*NB_OUT_X*NB_OUT_Y] = result[i][j][k];

}
