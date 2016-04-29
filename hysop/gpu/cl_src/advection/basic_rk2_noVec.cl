/**
 * @file advection/basic_noVec.cl
 * Advection function, basic version
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
 * @param gvelo Global velocity field.
 * @return Particle position
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 */
float advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh)
{
  float v, 			/* Velocity at point */
    vp,				/* Velocity at right point */
    p,				/* Normalized intermediary position */
    c = i * mesh->dx.x + mesh->min_position,  /* initial coordinate */
    hdt = 0.5 * dt;		/* half time step */
  int i_ind,			/* Interpolation left point */
    i_ind_p;			/* Interpolation right point */

#if !(ADVEC_IS_MULTISCALE)
  // single-scale:
  v = velocity_cache[noBC_id(i)]; 	/* k = k1 */
#else
  // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
  p = c * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = i_ind + V_GHOSTS_NB;
  i_ind_p = i_ind + 1;
  v = mix(velocity_cache[noBC_id(i_ind)],
	  velocity_cache[noBC_id(i_ind_p)],p);
#endif

  p = (c + hdt*v) * mesh->v_invdx;

  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);

  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);

  v = velocity_cache[noBC_id(i_ind)];
  vp = velocity_cache[noBC_id(i_ind_p)];
  v = (p*(vp-v) + v);

  return c + dt * v;
}
/* Operations number :  */
/*   - 2 positions = 2 * 2 */
/*   - 1 iterpolation = 9 */
/* Total = 13 */
