/**
 * @file builtin_noVec.cl
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
 * @param velocity_cache Local velocity field.
 * @return Particle position
 *
 * @remark NB_I, NB_II, NB_III : points number in directions from 1st varying index to last.
 */
float advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh)
{
  float v, 			/* Velocity at point */
    c = fma(i, mesh->dx.x, mesh->min_position);	/* initial coordinate */

#if !(ADVEC_IS_MULTISCALE)
  // single-scale:
  v = velocity_cache[noBC_id(i)];
#else
  float p;
  int i_ind,			/* Interpolation left point */
    i_ind_p;			/* Interpolation right point */
  // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
  p = c * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = i_ind + V_GHOSTS_NB;
  i_ind_p = i_ind + 1;
  v = mix(velocity_cache[noBC_id(i_ind)],
	  velocity_cache[noBC_id(i_ind_p)],p);
#endif

  return fma(dt, v, c);
}
/* Operations number :  */
/*   - 3 positions = 3 * fma */
/*   - 1 iterpolation = 2 + 1 * mix */
/*   - dt/2 = 1 */
/* 1mix <=> 3flop : mix(x,y,a) = x+(y-x)*a */
/* Total = 3 fma + 1 mix + 3 = 12flop */
