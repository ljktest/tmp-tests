/**
 * @file advection/comm_basic_noVec.cl
 * Advection function, basic version, mpi communications on the host side
 */

float advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh);


/**
 * Compute the position of a particle with a RK2 integration scheme. Velocity is linearly interpolated from the global field.
 * Use of builtin OpenCL functions fma and mix.
 *
 * @param i Particle index.
 * @param dt Time step.
 * @param dx Space step.
 * @param invdx 1/dx.
 * @param velocity_cache Local velocity field.
 * @return Particle position
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 * @remark T_NB_I: global points number in the 1st direction (mpi cutted direction)
 * @remark START_INDEX Global staring index for computational points
 * @remark STOP_INDEX Global stop index for computational points
 */
float advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh)
{
  float v, 			/* Velocity at point */
    vp,				/* Velocity at right point */
    p,				/* Normalized intermediary position */
    c = i * mesh->dx.x + mesh->min_position, /* initial coordinate */
    hdt = 0.5 * dt;		/* half time step */
  int i_ind,			/* Interpolation left point */
    i_ind_p;			/* Interpolation right point */

#if (V_NB_I-2*V_GHOSTS_NB) == NB_I
  // single-scale:
  v = velocity_cache[noBC_id(i + V_GHOSTS_NB)]; 	/* k = k1 */
#else
  // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
  p = c * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = i_ind - (V_START_INDEX-V_GHOSTS_NB);
  i_ind_p = i_ind + 1;
  v = mix(velocity_cache[noBC_id(i_ind)],
	  velocity_cache[noBC_id(i_ind_p)],p);
#endif
  p = (c + hdt*v) * mesh->v_invdx;

  i_ind = convert_int_rtn(p);
  if( i_ind>=(V_START_INDEX-MS_INTERPOL_SHIFT) && i_ind < (V_STOP_INDEX-V_GHOSTS_NB))
    {
      p = p - convert_float(i_ind);

      i_ind = i_ind - (V_START_INDEX-V_GHOSTS_NB);
      i_ind_p = i_ind + 1;

      v = velocity_cache[noBC_id(i_ind)];
      vp = velocity_cache[noBC_id(i_ind_p)];
      v = (p*(vp-v) + v);

      p = c + dt * v;
    }
  else
    {
      p = (1000*T_NB_I)*1.0 + p;
    }

  return p;
}
/* Operations number :  */
/*   - 2 positions = 2 * 2 */
/*   - 1 iterpolation = 9 */
/* Total = 13 */
