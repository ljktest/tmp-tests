/**
 * @file basic_rk4_noVec.cl
 * Advection function (RK4 scheme), basic version
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
    vp,				/* Velocity at right point */
    p,				/* Intermediary position */
    k,				/* rk averaged velocity */
    kn,				/* rk intermediate velocity */
    c = i * mesh->dx.x + mesh->min_position,  /* initial coordinate */
    hdt = 0.5 * dt;		/* half time step */
  int i_ind,			/* Interpolation left point */
    i_ind_p;			/* Interpolation right point */

  //k1 = f(t,y)
  //k2 = f(t + dt/2, y + dt/2 * k1)
  //k3 = f(t + dt/2, y + dt/2 * k2)
  //k4 = f(t + dt, y + dt * k3)
  //result = y + dt/6( k1 + 2 * k2 + 2 * k3 + k4)

#if !(ADVEC_IS_MULTISCALE)
  // single-scale:
  k = velocity_cache[noBC_id(i)]; 	/* k = k1 */
#else
  // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
  p = c * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = i_ind + V_GHOSTS_NB;
  i_ind_p = i_ind + 1;
  k = mix(velocity_cache[noBC_id(i_ind)],
	  velocity_cache[noBC_id(i_ind_p)],p);
#endif

  p = (c + hdt * k) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  v = velocity_cache[noBC_id(i_ind)];
  vp = velocity_cache[noBC_id(i_ind_p)];
  kn = p*(vp-v) + v;		/* kn = k2 */

  k += 2.0 * kn;		/* k = k1 + 2*k2 */

  p = (c + hdt * kn) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  v = velocity_cache[noBC_id(i_ind)];
  vp = velocity_cache[noBC_id(i_ind_p)];
  kn = p*(vp-v) + v;		/* kn = k3 */

  k += 2.0 * kn;		/* k = k1 + 2*k2 + 2*k3 */

  p = (c + dt * kn) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  v = velocity_cache[noBC_id(i_ind)];
  vp = velocity_cache[noBC_id(i_ind_p)];
  kn = p*(vp-v) + v;		/* kn = k4 */

  k += kn;			/* k = k1 + 2*k2 + 2*k3 + k4 */

  return c + dt * k*0.16666666666666666;
}
/* Operations number :  */
/*   - 4 positions = 4 * 2 + 3 */
/*   - 3 iterpolation = 3 * 9 */
/*   - velocity weights = 5*/
/* Total = 41 */
