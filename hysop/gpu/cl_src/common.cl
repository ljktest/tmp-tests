/**
 * @file common.cl
 * Common parameters for advection and remeshing kernels.
 */

inline uint noBC_id(int id);

#ifdef WITH_NOBC
/**
 * Mapping to local memory arrays to avoir banck conflics.
 * 1D buffer is taken as 2D one with wor-items vs. particles.
 *
 * @param id 1D index
 *
 * @return 2D index
 */
inline uint noBC_id(int id){
  return (id%PART_NB_PER_WI)*WI_NB+(id/PART_NB_PER_WI);
}
#else
/**
 * Leave mapping unchanged, 1D.
 *
 * @param id 1D index
 *
 * @return 1D index
 */
inline uint noBC_id(int id){
  return id;
}
#endif

/**
 * Constants for remeshing formulas:
 *   - L2_1 1
 *   - L2_2 2
 *   - L2_3 3
 *   - L2_4 4
 *   - L4_2 5
 *   - L4_3 6
 *   - L4_4 7
 *   - L6_3 8
 *   - L6_4 9
 *   - L6_5 10
 *   - L6_6 11
 *   - L8_4 12
 *   - M8PRIME 13
 */
#define L2_1 1
#define L2_2 2
#define L2_3 3
#define L2_4 4
#define L4_2 5
#define L4_3 6
#define L4_4 7
#define L6_3 8
#define L6_4 9
#define L6_5 10
#define L6_6 11
#define L8_4 12
#define M8PRIME 13
#define LINEAR 14

/**
 * Remeshing configuration
 */
#if FORMULA == L2_1
#define REMESH_SHIFT 1
#define REMESH(greek) greek##_l2_1
#elif FORMULA == L2_2
#define REMESH_SHIFT 1
#define REMESH(greek) greek##_l2_2
#elif FORMULA == L2_3
#define REMESH_SHIFT 1
#define REMESH(greek) greek##_l2_3
#elif FORMULA == L2_4
#define REMESH_SHIFT 1
#define REMESH(greek) greek##_l2_4

#elif FORMULA == L4_2
#define REMESH_SHIFT 2
#define REMESH(greek) greek##_l4_2
#elif FORMULA == L4_3
#define REMESH_SHIFT 2
#define REMESH(greek) greek##_l4_3
#elif FORMULA == L4_4
#define REMESH_SHIFT 2
#define REMESH(greek) greek##_l4_4

#elif FORMULA == M8PRIME
#define REMESH_SHIFT 3
#define REMESH(greek) greek##_M8p
#elif FORMULA == L6_3
#define REMESH_SHIFT 3
#define REMESH(greek) greek##_l6_3
#elif FORMULA == L6_4
#define REMESH_SHIFT 3
#define REMESH(greek) greek##_l6_4
#elif FORMULA == L6_5
#define REMESH_SHIFT 3
#define REMESH(greek) greek##_l6_5
#elif FORMULA == L6_6
#define REMESH_SHIFT 3
#define REMESH(greek) greek##_l6_6

#elif FORMULA == L8_4
#define REMESH_SHIFT 4
#define REMESH(greek) greek##_l8_4
#endif


/**
 * Multi-scale configuration
 */
#define ADVEC_IS_MULTISCALE (V_NB_I-2*V_GHOSTS_NB) != NB_I
#if MS_FORMULA == LINEAR
#define MS_INTERPOL_SHIFT 0
// MS_INTERPOL not used
#elif MS_FORMULA == L2_1
#define MS_INTERPOL_SHIFT 1
#define MS_INTERPOL(greek) greek##_l2_1
#elif MS_FORMULA == L4_2
#define MS_INTERPOL_SHIFT 2
#define MS_INTERPOL(greek) greek##_l4_2
#elif MS_FORMULA == L4_4
#define MS_INTERPOL_SHIFT 2
#define MS_INTERPOL(greek) greek##_l4_4
#else
//Default case for single-scale (only used in comm advection)
#define MS_INTERPOL_SHIFT 0
#endif

/*
a minmax element is a 12 int defined as follows:
*/
#define L_MIN_X 0
#define L_MAX_X 1
#define L_MIN_Y 2
#define L_MAX_Y 3
#define L_MIN_Z 4
#define L_MAX_Z 5
#define R_MIN_X 6
#define R_MAX_X 7
#define R_MIN_Y 8
#define R_MAX_Y 9
#define R_MIN_Z 10
#define R_MAX_Z 11

/* Structure to store __constants advection parameters */
typedef struct AdvectionMeshInfo
{
  float4 dx;                   /* Mesh step (advected grid) */
  float4 v_dx;                 /* Mesh step (velocity) */
  float min_position;           /* Domain minimum coordinate in current direction */
  float invdx;                 /* Store 1./dx.x */
  float v_invdx;               /* Store 1./v_dx.x */
  float x;                     /* Padding */
} AdvectionMeshInfo;

/* Finite differences constants */
#define FD_C_2 88
#define FD_C_4 99
