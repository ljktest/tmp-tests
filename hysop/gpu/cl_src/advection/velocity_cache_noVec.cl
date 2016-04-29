void fill_velocity_cache(__global const float* gvelo,
			 uint gidX, uint gidY, uint gidZ,
			 __local float* gvelo_loc,
#if ADVEC_IS_MULTISCALE
			 float inv_v_dx_y, float inv_v_dx_z,
#endif
			 __constant struct AdvectionMeshInfo* mesh);

void fill_velocity_cache(__global const float* gvelo,
			 uint gidX, uint gidY, uint gidZ,
			 __local float* gvelo_loc,
#if ADVEC_IS_MULTISCALE
			 float inv_v_dx_y, float inv_v_dx_z,
#endif
			 __constant struct AdvectionMeshInfo* mesh)
{
  uint i;

  // ********************************
  // **    Single Scale
  // ********************************
#if !(ADVEC_IS_MULTISCALE)
  // Single scale : Velocity and scalar grids are identical : cache is just read from global
  uint line_index = gidY*V_NB_I + gidZ*V_NB_I*V_NB_II; /* Current 1D problem index */
  for(i=gidX; i<V_NB_I; i+=(WI_NB))
    {
      /* Read velocity */
      /* Fill velocity cache */
      gvelo_loc[noBC_id(i)] = gvelo[i+line_index];
    }

  // ********************************
  // **    Multi-Scale
  // ********************************
  // Velocity cache is interpolated from global memory
#else


#if NB_III == 1
  //  Multi-Scale (2D)

  float line_posY, hY;
  int indY;
#if MS_FORMULA == LINEAR
  int2 v_line_index;
  float2 wY;
#elif MS_FORMULA == L2_1
  int4 v_line_index;
  float4 wY;
#elif MS_FORMULA == L4_2 ||  MS_FORMULA == L4_4
  // Only the 6 first elements will be used
  int8 v_line_index;
  float8 wY;
#endif

  line_posY = (gidY * mesh->dx.y) * inv_v_dx_y; // mesh->v_dx.y;
  indY = convert_int_rtn(line_posY);
  hY = line_posY - convert_float(indY);


#if MS_FORMULA == LINEAR
  wY.s1 = hY;
  wY.s0 = 1.0 - wY.s1;
#else
  wY.s0 = MS_INTERPOL(alpha)(hY);
  wY.s1 = MS_INTERPOL(beta)(hY);
  wY.s2 = MS_INTERPOL(gamma)(hY);
#if MS_INTERPOL_SHIFT > 1
  wY.s3 = MS_INTERPOL(delta)(hY);
  wY.s4 = MS_INTERPOL(eta)(hY);
  wY.s5 = 1.0 - wY.s0 - wY.s1 - wY.s2 - wY.s3 - wY.s4;
#else
  wY.s3 = 1.0 - wY.s0 - wY.s1 - wY.s2;
#endif
#endif

  indY = indY + V_GHOSTS_NB - MS_INTERPOL_SHIFT;

  v_line_index.s0 = indY * V_NB_I;
  v_line_index.s1 = (indY + 1) * V_NB_I;
#if MS_INTERPOL_SHIFT > 0
  v_line_index.s2 = (indY + 2) * V_NB_I;
  v_line_index.s3 = (indY + 3) * V_NB_I;
#elif MS_INTERPOL_SHIFT > 1
  v_line_index.s4 = (indY + 4) * V_NB_I;
  v_line_index.s5 = (indY + 5) * V_NB_I;
#endif


  for(i=gidX; i<V_NB_I; i+=(WI_NB)){
    gvelo_loc[noBC_id(i)] = wY.s0 * gvelo[i + v_line_index.s0];
    gvelo_loc[noBC_id(i)] += wY.s1 * gvelo[i + v_line_index.s1];
#if MS_INTERPOL_SHIFT > 0
    gvelo_loc[noBC_id(i)] += wY.s2 * gvelo[i + v_line_index.s2];
    gvelo_loc[noBC_id(i)] += wY.s3 * gvelo[i + v_line_index.s3];
#elif MS_INTERPOL_SHIFT > 1
    gvelo_loc[noBC_id(i)] += wY.s4 * gvelo[i + v_line_index.s4];
    gvelo_loc[noBC_id(i)] += wY.s5 * gvelo[i + v_line_index.s5];
#endif
  }
			 /* nombre d'opérations 2D Linéaire:
			    - calcul des poids de ligne : 4flop (par wi)
			    - calcul de la vitesse : 3flop par point de grille de vitesse
			 */


#else
  //  Multi-Scale (3D)

  float line_posY, hY;
  float line_posZ, hZ;
  int indY, indZ;
#if MS_FORMULA == LINEAR
  int2 v_line_indexY, v_line_indexZ;
  float2 wY, wZ;
#elif MS_FORMULA == L2_1
  int4 v_line_indexY, v_line_indexZ;
  float4 wY, wZ;
#elif MS_FORMULA == L4_2 || MS_FORMULA == L4_4
  int8 v_line_indexY, v_line_indexZ;
  float8 wY, wZ;
#endif

  line_posY = (gidY * mesh->dx.y) * inv_v_dx_y; // mesh->v_dx.y;
  line_posZ = (gidZ * mesh->dx.z) * inv_v_dx_z;// mesh->v_dx.z;
  indY = convert_int_rtn(line_posY);
  indZ = convert_int_rtn(line_posZ);
  hY = line_posY - convert_float(indY);
  hZ = line_posZ - convert_float(indZ);

#if MS_FORMULA == LINEAR
  wY.s1 = hY;
  wY.s0 = 1.0 - wY.s1;
  wZ.s1 = hZ;
  wZ.s0 = 1.0 - wZ.s1;
#else
  wY.s0 = MS_INTERPOL(alpha)(hY);
  wY.s1 = MS_INTERPOL(beta)(hY);
  wY.s2 = MS_INTERPOL(gamma)(hY);
  wZ.s0 = MS_INTERPOL(alpha)(hZ);
  wZ.s1 = MS_INTERPOL(beta)(hZ);
  wZ.s2 = MS_INTERPOL(gamma)(hZ);
#if MS_INTERPOL_SHIFT > 1
  wY.s3 = MS_INTERPOL(delta)(hY);
  wY.s4 = MS_INTERPOL(eta)(hY);
  wY.s5 = 1.0 - wY.s0 - wY.s1 - wY.s2 - wY.s3 - wY.s4;
  wZ.s3 = MS_INTERPOL(delta)(hZ);
  wZ.s4 = MS_INTERPOL(eta)(hZ);
  wZ.s5 = 1.0 - wZ.s0 - wZ.s1 - wZ.s2 - wZ.s3 - wZ.s4;
#else
  wY.s3 = 1.0 - wY.s0 - wY.s1 - wY.s2;
  wZ.s3 = 1.0 - wZ.s0 - wZ.s1 - wZ.s2;
#endif
#endif

  indY = indY + V_GHOSTS_NB - MS_INTERPOL_SHIFT;
  indZ = indZ + V_GHOSTS_NB - MS_INTERPOL_SHIFT;

  v_line_indexY.s0 = indY * V_NB_I;
  v_line_indexY.s1 = (indY + 1) * V_NB_I;
  v_line_indexZ.s0 = indZ * V_NB_I * V_NB_II;
  v_line_indexZ.s1 = (indZ + 1) * V_NB_I * V_NB_II;
#if MS_INTERPOL_SHIFT > 0
  v_line_indexY.s2 = (indY + 2) * V_NB_I;
  v_line_indexY.s3 = (indY + 3) * V_NB_I;
  v_line_indexZ.s2 = (indZ + 2) * V_NB_I * V_NB_II;
  v_line_indexZ.s3 = (indZ + 3) * V_NB_I * V_NB_II;
#elif MS_INTERPOL_SHIFT > 1
  v_line_indexY.s4 = (indY + 4) * V_NB_I;
  v_line_indexY.s5 = (indY + 5) * V_NB_I;
  v_line_indexZ.s4 = (indZ + 4) * V_NB_I * V_NB_II;
  v_line_indexZ.s5 = (indZ + 5) * V_NB_I * V_NB_II;
#endif

  for(i=gidX; i<V_NB_I; i+=(WI_NB)){
    gvelo_loc[noBC_id(i)] = wY.s0 * wZ.s0 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s0 * wZ.s1 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s1];
    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s0 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s1 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s1];
#if MS_INTERPOL_SHIFT > 0
    gvelo_loc[noBC_id(i)] += wY.s0 * wZ.s2 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s0 * wZ.s3 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s3];

    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s2 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s3 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s3];

    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s0 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s1 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s1];
    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s2 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s3 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s3];

    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s0 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s1 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s1];
    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s2 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s3 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s3];
#elif MS_INTERPOL_SHIFT > 1
    gvelo_loc[noBC_id(i)] += wY.s0 * wZ.s4 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s0 * wZ.s5 * gvelo[i + v_line_indexY.s0 + v_line_indexZ.s5];

    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s4 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s1 * wZ.s5 * gvelo[i + v_line_indexY.s1 + v_line_indexZ.s5];

    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s4 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s2 * wZ.s5 * gvelo[i + v_line_indexY.s2 + v_line_indexZ.s5];

    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s4 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s3 * wZ.s5 * gvelo[i + v_line_indexY.s3 + v_line_indexZ.s5];

    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s0 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s1 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s1];
    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s2 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s3 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s3];
    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s4 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s4 * wZ.s5 * gvelo[i + v_line_indexY.s4 + v_line_indexZ.s5];

    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s0 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s0];
    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s1 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s1];
    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s2 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s2];
    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s3 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s3];
    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s4 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s4];
    gvelo_loc[noBC_id(i)] += wY.s5 * wZ.s5 * gvelo[i + v_line_indexY.s5 + v_line_indexZ.s5];
#endif
  }
			 /* nombre d'opérations 3D Linéaire:
			    - calcul des poids de ligne : 8flop (par wi)
			    - calcul de la vitesse : 11flop par point de grille de vitesse
			 */

#endif
#endif
}
