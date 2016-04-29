/**
 * @file builtin.cl
 * Advection function, vectorized version.
 */

float__N__ advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh);


/**
 * Compute the position of a particle with a RK2 integration scheme. Velocity is linearly interpolated from the global field.
 * Use of builtin OpenCL functions fma and mix. Computations through OpenCL vector types.
 *
 * @param i Particle index.
 * @param dt Time step.
 * @param dx Space step.
 * @param invdx 1/dx.
 * @param velocity_cache Local velocity cache.
 * @return Particle position.
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @see hysop.gpu.tools.parse_file
 */
float__N__ advection(uint i, float dt, __local float* velocity_cache, __constant struct AdvectionMeshInfo* mesh)
{
  float__N__ v,        		/* Velocity at point */
    vp,				/* Velocity at right point */
    p,				/* Intermediary position */
    c,				/* initial coordinate */
    hdt = (float__N__)(0.5*dt);	/* half time step */
  int__N__ i_ind,		/* Interpolation left point */
    i_ind_p;			/* Interpolation right point */

  c = (float__N__)((i+__NN__)*mesh->dx.x,
		       );
  c = c + mesh->min_position;

#if !(ADVEC_IS_MULTISCALE)
  // single-scale:
  v = (float__N__)(velocity_cache[noBC_id(i+__NN__)],
		   );
  p = fma(hdt, v, c) * mesh->v_invdx;
#else
  // multi-scale : interpolate v from velocity buffer (of length V_NB_I)
  p = c * mesh->v_invdx;
  i_ind = convert_int__N___rtn(p);
  p = p - convert_float__N__(i_ind);
  i_ind = i_ind + V_GHOSTS_NB;
  i_ind_p = i_ind + 1;
  v = (float__N__)(velocity_cache[noBC_id(i_ind.s__NN__)],
		   );
  vp = (float__N__)(velocity_cache[noBC_id(i_ind_p.s__NN__)],
		    );
  p = fma(hdt, mix(v,vp,p), c) * v_invdx;
#endif

  i_ind = convert_int__N___rtn(p);
  p = p - convert_float__N__(i_ind);

  i_ind = ((i_ind + V_GHOSTS_NB + V_NB_I) % V_NB_I);
  i_ind_p = ((i_ind + 1) % V_NB_I);
  v = (float__N__)(velocity_cache[noBC_id(i_ind.s__NN__)],
		   );
  vp = (float__N__)(velocity_cache[noBC_id(i_ind_p.s__NN__)],
		    );
  return fma(mix(v,vp,p),dt,c);
}
