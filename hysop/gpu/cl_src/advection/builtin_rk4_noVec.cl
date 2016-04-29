/**
 * @file builtin_rk4_noVec.cl
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
  float p,		       /* Intermediary position */
    k,			       /* rk averaged velocity */
    kn,			       /* rk intermediate velocity */
    c = fma(i, mesh->dx.x, mesh->min_position), /* initial coordinate */
    hdt = 0.5 * dt;	       /* half time step */
  int i_ind,		       /* Interpolation left point */
    i_ind_p;		       /* Interpolation right point */

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

  p = fma(hdt, k, c) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  kn = mix(velocity_cache[noBC_id(i_ind)],
	   velocity_cache[noBC_id(i_ind_p)],p);		/* kn = k2 */

  k += 2.0 * kn;		/* k = k1 + 2*k2 */

  p = fma(hdt, kn, c) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  kn = mix(velocity_cache[noBC_id(i_ind)],
	   velocity_cache[noBC_id(i_ind_p)],p);		/* kn = k3 */

  k += 2.0 * kn;		/* k = k1 + 2*k2 + 2*k3 */

  p = fma(dt, kn, c) * mesh->v_invdx;
  i_ind = convert_int_rtn(p);
  p = p - convert_float(i_ind);
  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  kn = mix(velocity_cache[noBC_id(i_ind)],
	   velocity_cache[noBC_id(i_ind_p)],p);		/* kn = k4 */

  k += kn;			/* k = k1 + 2*k2 + 2*k3 + k4 */

  return fma(k, dt*0.16666666666666666, c);
}

/* Operations number :  */
/*   - 5 positions = 5 * fma*/
/*   - 3 iterpolation = 3 * (1 * mix + 2) */
/*   - velocity weights = 7 */
/*   - dt/2, dt/6 = 2 */
/* Total = 5 fma + 3 mix + 13 = 32flop */
