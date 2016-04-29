/**
 * @file private.cl
 * Remeshing function, vectorized, private variable.
 */

void remesh(uint i, __RCOMP_P float s__ID__, float p, __RCOMP_P__local float* gscal_loc__ID__, __constant struct AdvectionMeshInfo* mesh);


/**
 * Remesh particles in local buffer.
 *
 * Remeshing formula is given a compiling time.
 * Use of builtin OpenCL functions fma and mix. Computations through OpenCL vector types.
 * Use of a private temporary variable for remeshing weights.
 *
 * @param i Particle index
 * @param dx Space step
 * @param invdx 1/dx
 * @param s Particle scalar
 * @param p Particle position
 * @param gscal_loc Local buffer for result
 *
 * @remark <code>NB_I</code>, <code>NB_II</code>, <code>NB_III</code> : points number in directions from 1st varying index to last.
 * @remark <code>__N__</code> is expanded at compilation time by vector width.
 * @remark <code>__NN__</code> is expanded at compilation time by a sequence of integer for each vector component.
 * @remark <code>FORMULA</code> : remeshing formula flag {<code>M4PRIME</code>, <code>M6PRIME</code>, <code>M8PRIME</code>, <code>L6STAR</code>}
 * @remark <code>__RCOMP_I</code> flag is for instruction expansion for the different remeshed components.
 * @remark <code>__RCOMP_P</code> flag is for function parameter expansion for the different remeshed components.
 * @remark <code>__ID__</code> is replaced by the remeshed component id in an expansion.
 * @remark <code>REMESH</code> is a function-like macro expanding to the proper remeshing formula (i.e.: <code>REMESH(alpha)</code> -> <code>alpha_l2_1</code>)
 * @see hysop.gpu.tools.parse_file
 * @see hysop.gpu.cl_src.common
 */
void remesh(uint i,
	    __RCOMP_P float s__ID__,
	    float p,
	    __RCOMP_P__local float* gscal_loc__ID__,
	    __constant struct AdvectionMeshInfo* mesh){
  float y,			/* Normalized distance to nearest left grid point */
    w;			/* Temporary remeshing weights */
  __RCOMP_I float temp__ID__;
  int ind;			/* Integer coordinate */
  uint index;		/* Remeshing index */

  p = p - mesh->min_position;

  ind = convert_int_rtn(p * mesh->invdx);
  y = (p - convert_float(ind) * mesh->dx.x) * mesh->invdx;

  index = convert_uint((ind - REMESH_SHIFT + NB_I) % NB_I);

  w = REMESH(alpha)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(beta)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(gamma)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(delta)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

#if REMESH_SHIFT > 1
  index = (index + 1) % NB_I;
  w = REMESH(eta)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(zeta)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 2
  index = (index + 1) % NB_I;
  w = REMESH(theta)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(iota)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);
#endif

#if REMESH_SHIFT > 3
  index = (index + 1) % NB_I;
  w = REMESH(kappa)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);

  index = (index + 1) % NB_I;
  w = REMESH(mu)(y);
  __RCOMP_Itemp__ID__ = w * s__ID__;
  __RCOMP_Igscal_loc__ID__[noBC_id(index)] += temp__ID__;
  barrier(CLK_LOCAL_MEM_FENCE);
  #endif
}
