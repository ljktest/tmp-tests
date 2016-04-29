/**
 * @file weights_noVec_builtin.cl
 * Remeshing formulas, vectorized version, use of builtin OpenCL fma.
 * Polynomials under Horner form.
 */

inline float alpha_l2_1(float y){
  return (y*fma(y,fma(y,-1.0, 2.0), - 1.0) * 0.5);}
inline float beta_l2_1(float y){
  return (fma(y*y, fma(y, 3.0, -5.0), 2.0) * 0.5);}
inline float gamma_l2_1(float   y){
  return ((y * fma(y , fma(-3.0, y, 4.0), 1.0)) * 0.5);}
inline float delta_l2_1(float y){
  return ((y * y * fma(1.0, y, - 1.0)) * 0.5);}


inline float alpha_l2_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, 2.0, -5.0), 3.0), 1.0), -1.0)) * 0.5);}
inline float beta_l2_2(float y){
  return (fma(y * y, fma(y, fma(y, fma(y, -6.0, 15.0), -9.0), -2.0), 2.0) * 0.5);}
inline float gamma_l2_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, 6.0, -15.0), 9.0), 1.0), 1.0)) * 0.5);}
inline float delta_l2_2(float y){
  return ((y * y * y * fma(y, fma(y, -2.0, 5.0), -3.0)) * 0.5);}


inline float alpha_l2_3(float y){
  return ((y * fma(y, fma(y * y, fma(y, fma(y, fma(y, -6.0, 21.0), -25.0), 10.0), 1.0), -1.0)) * 0.5);}
inline float beta_l2_3(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, 18.0, -63.0), 75.0), -30.0), -2.0), 2.0) * 0.5);}
inline float gamma_l2_3(float y){
  return ((y * fma(y, fma(y * y, fma(y, fma(y, fma(y, -18.0, 63.0), -75.0), 30.0), 1.0), 1.0)) * 0.5);}
inline float delta_l2_3(float y){
  return ((y * y * y * y * fma(y, fma(y, fma(y, 6.0, -21.0), 25.0), -10.0)) * 0.5);}


inline float alpha_l2_4(float y){
  return ((y * fma(y, fma(y * y * y, fma(y, fma(y, fma(y, fma(y, 20.0, -90.0), 154.0), -119.0), 35.0), 1.0), -1.0)) * 0.5);}
inline float beta_l2_4(float y){
  return (fma(y * y, fma(y * y * y, fma(y, fma(y, fma(y, fma(y, -60.0, 270.0), -462.0), 357.0), -105.0), -2.0), 2.0) * 0.5);}
inline float gamma_l2_4(float y){
  return ((y * fma(y, fma(y * y * y, fma(y, fma(y, fma(y, fma(y, 60.0, -270.0), 462.0), -357.0), 105.0), 1.0), 1.0)) * 0.5);}
inline float delta_l2_4(float y){
  return ((y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, -20.0, 90.0), -154.0), 119.0), -35.0)) * 0.5);}


inline float alpha_l4_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, -5.0, 13.0), -9.0), -1.0), 2.0)) * 0.041666666666666664);}
inline float beta_l4_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, 25.0, -64.0), 39.0), 16.0), -16.0)) * 0.041666666666666664);}
inline float gamma_l4_2(float y){
  return (fma(y * y, fma(y, fma(y, fma(y, -50.0, 126.0), -70.0), -30.0), 24.0) * 0.041666666666666664);}
inline float delta_l4_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, 50.0, -124.0), 66.0), 16.0), 16.0)) * 0.041666666666666664);}
inline float eta_l4_2(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, -25.0, 61.0), -33.0), -1.0), -2.0)) * 0.041666666666666664);}
inline float zeta_l4_2(float y){
  return ((y * y * y * fma(y, fma(y, 5.0, -12.0), 7.0)) * 0.041666666666666664);}


inline float alpha_l4_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 14.0, -49.0), 58.0), -22.0), -2.0), -1.0), 2.0)) * 0.041666666666666664);}
inline float beta_l4_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -70.0, 245.0), -290.0), 111.0), 4.0), 16.0), -16.0)) * 0.041666666666666664);}
inline float gamma_l4_3(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, 140.0, -490.0), 580.0), -224.0), -30.0), 24.0) * 0.041666666666666664);}
inline float delta_l4_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -140.0, 490.0), -580.0), 226.0), -4.0), 16.0), 16.0)) * 0.041666666666666664);}
inline float eta_l4_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 70.0, -245.0), 290.0), -114.0), 2.0), -1.0), -2.0)) * 0.041666666666666664);}
inline float zeta_l4_3(float y){
  return ((y * y * y * y * fma(y, fma(y, fma(y, -14.0, 49.0), -58.0), 23.0)) * 0.041666666666666664);}


inline float alpha_l4_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -46.0, 207.0), -354.0), 273.0), -80.0), 1.0), -2.0), -1.0), 2.0)) * 0.041666666666666664);}
inline float beta_l4_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 230.0, -1035.0), 1770.0), -1365.0), 400.0), -4.0), 4.0), 16.0), -16.0)) * 0.041666666666666664);}
inline float gamma_l4_4(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, fma(y, fma(y, -460.0, 2070.0), -3540.0), 2730.0), -800.0), 6.0), -30.0), 24.0) * 0.041666666666666664);}
inline float delta_l4_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 460.0, -2070.0), 3540.0), -2730.0), 800.0), -4.0), -4.0), 16.0), 16.0)) * 0.041666666666666664);}
inline float eta_l4_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -230.0, 1035.0), -1770.0), 1365.0), -400.0), 1.0), 2.0), -1.0), -2.0)) * 0.041666666666666664);}
inline float zeta_l4_4(float y){
  return ((y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, 46.0, -207.0), 354.0), -273.0), 80.0)) * 0.041666666666666664);}


inline float alpha_M8p(float y){
  return (fma(y,fma(y,fma(y,fma(y,fma(y,fma(y,fma(-10.0,y, + 21.0), + 28.0), - 105.0), + 70.0), + 35.0), - 56.0), + 17.0) * 0.00029761904761904765);}
inline float beta_M8p(float y){
  return (fma(y,fma(y,fma(y,fma(y,fma(y,fma(y,fma(70.0,y, - 175.0), - 140.0), + 770.0), - 560.0), - 350.0), + 504.0), - 102.0) * 0.00029761904761904765);}
inline float gamma_M8p(float y){
  return (fma(y,fma(y,fma(y,fma(y,fma(y,fma(y,fma(-210.0,y, + 609.0), + 224.0), - 2135.0), + 910.0), + 2765.0), - 2520.0), + 255.0) * 0.00029761904761904765);}
inline float delta_M8p(float y){
  return (fma(y*y, fma(y*y, fma(y*y, fma(70.0,y, - 231.0), + 588.0), - 980.0), + 604.0) * 0.001488095238095238);}
inline float eta_M8p(float y){
  return (fma(y,fma(y,fma(y,fma(y,fma(y,fma(y,fma(-70.0,y, 259.0), - 84.0), - 427.0), - 182.0), + 553.0), + 504.0), + 51.0) * 0.001488095238095238);}
inline float zeta_M8p(float y){
  return (fma(y,fma(y,fma(y,fma(y,fma(y,fma(y,fma(210.0,y,- 861.0), + 532.0), + 770.0), + 560.0), - 350.0), - 504.0), - 102.0) * 0.00029761904761904765);}
inline float theta_M8p(float y){
  return (fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(-70.0, y, 315.0), -280.0), -105.0), -70.0), 35.0), 56.0), 17.0) * 0.00029761904761904765);}
inline float iota_M8p(float y){
  return ((y * y * y * y * y * fma(y , fma(10.0 , y ,- 49.0) , 56.0)) * 0.00029761904761904765);}


inline float alpha_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -89.0, 312.0), -370.0), 140.0), 15.0), 4.0), -12.0)) * 0.001388888888888889);}
inline float beta_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 623.0, -2183.0), 2581.0), -955.0), -120.0), -54.0), 108.0)) * 0.001388888888888889);}
inline float gamma_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -1869.0, 6546.0), -7722.0), 2850.0), 195.0), 540.0), -540.0)) * 0.001388888888888889);}
inline float delta_l6_3(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, 3115.0, -10905.0), 12845.0), -4795.0), -980.0), 720.0) * 0.001388888888888889);}
inline float eta_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -3115.0, 10900.0), -12830.0), 4880.0), -195.0), 540.0), 540.0)) * 0.001388888888888889);}
inline float zeta_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 1869.0, -6537.0), 7695.0), -2985.0), 120.0), -54.0), -108.0)) * 0.001388888888888889);}
inline float theta_l6_3(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -623.0, 2178.0), -2566.0), 1010.0), -15.0), 4.0), 12.0)) * 0.001388888888888889);}
inline float iota_l6_3(float y){
  return ((y * y * y * y * fma(y, fma(y, fma(y, 89.0, -311.0), 367.0), -145.0)) * 0.001388888888888889);}


inline float alpha_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 290.0, -1305.0), 2231.0), -1718.0), 500.0), -5.0), 15.0), 4.0), -12.0)) * 0.001388888888888889);}
inline float beta_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -2030.0, 9135.0), -15617.0), 12027.0), -3509.0), 60.0), -120.0), -54.0), 108.0)) * 0.001388888888888889);}
inline float gamma_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 6090.0, -27405.0), 46851.0), -36084.0), 10548.0), -195.0), 195.0), 540.0), -540.0)) * 0.001388888888888889);}
inline float delta_l6_4(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, fma(y, fma(y, -10150.0, 45675.0), -78085.0), 60145.0), -17605.0), 280.0), -980.0), 720.0) * 0.001388888888888889);}
inline float eta_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 10150.0, -45675.0), 78085.0), -60150.0), 17620.0), -195.0), -195.0), 540.0), 540.0)) * 0.001388888888888889);}
inline float zeta_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -6090.0, 27405.0), -46851.0), 36093.0), -10575.0), 60.0), 120.0), -54.0), -108.0)) * 0.001388888888888889);}
inline float theta_l6_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 2030.0, -9135.0), 15617.0), -12032.0), 3524.0), -5.0), -15.0), 4.0), 12.0)) * 0.001388888888888889);}
inline float iota_l6_4(float y){
  return ((y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, -290.0, 1305.0), -2231.0), 1719.0), -503.0)) * 0.001388888888888889);}


inline float alpha_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -1006.0, 5533.0), -12285.0), 13785.0), -7829.0), 1803.0), -3.0), -5.0), 15.0), 4.0), -12.0)) * 0.001388888888888889);}
inline float beta_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 7042.0, -38731.0), 85995.0), -96495.0), 54803.0), -12620.0), 12.0), 60.0), -120.0), -54.0), 108.0)) * 0.001388888888888889);}
inline float gamma_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -21126.0, 116193.0), -257985.0), 289485.0), -164409.0), 37857.0), -15.0), -195.0), 195.0), 540.0), -540.0)) * 0.001388888888888889);}
inline float delta_l6_5(float y){
  return (fma(y * y, fma(y * y, fma(y * y, fma(y, fma(y, fma(y, fma(y, fma(y, 35210.0, -193655.0), 429975.0), -482475.0), 274015.0), -63090.0), 280.0), -980.0), 720.0) * 0.001388888888888889);}
inline float eta_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -35210.0, 193655.0), -429975.0), 482475.0), -274015.0), 63085.0), 15.0), -195.0), -195.0), 540.0), 540.0)) * 0.001388888888888889);}
inline float zeta_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 21126.0, -116193.0), 257985.0), -289485.0), 164409.0), -37848.0), -12.0), 60.0), 120.0), -54.0), -108.0)) * 0.001388888888888889);}
inline float theta_l6_5(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -7042.0, 38731.0), -85995.0), 96495.0), -54803.0), 12615.0), 3.0), -5.0), -15.0), 4.0), 12.0)) * 0.001388888888888889);}
inline float iota_l6_5(float y){
  return ((y * y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, fma(y, 1006.0, -5533.0), 12285.0), -13785.0), 7829.0), -1802.0)) * 0.001388888888888889);}


inline float alpha_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 3604.0, -23426.0), 63866.0), -93577.0), 77815.0), -34869.0), 6587.0), 1.0), -3.0), -5.0), 15.0), 4.0), -12.0)) * 0.001388888888888889);}
inline float beta_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -25228.0, 163982.0), -447062.0), 655039.0), -544705.0), 244083.0), -46109.0), -6.0), 12.0), 60.0), -120.0), -54.0), 108.0)) * 0.001388888888888889);}
inline float gamma_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 75684.0, -491946.0), 1341186.0), -1965117.0), 1634115.0), -732249.0), 138327.0), 15.0), -15.0), -195.0), 195.0), 540.0), -540.0)) * 0.001388888888888889);}
inline float delta_l6_6(float y){
  return (fma(y * y, fma(y * y, fma(y * y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -126140.0, 819910.0), -2235310.0), 3275195.0), -2723525.0), 1220415.0), -230545.0), -20.0), 280.0), -980.0), 720.0) * 0.001388888888888889);}
inline float eta_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 126140.0, -819910.0), 2235310.0), -3275195.0), 2723525.0), -1220415.0), 230545.0), 15.0), 15.0), -195.0), -195.0), 540.0), 540.0)) * 0.001388888888888889);}
inline float zeta_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -75684.0, 491946.0), -1341186.0), 1965117.0), -1634115.0), 732249.0), -138327.0), -6.0), -12.0), 60.0), 120.0), -54.0), -108.0)) * 0.001388888888888889);}
inline float theta_l6_6(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 25228.0, -163982.0), 447062.0), -655039.0), 544705.0), -244083.0), 46109.0), 1.0), 3.0), -5.0), -15.0), 4.0), 12.0)) * 0.001388888888888889);}
inline float iota_l6_6(float y){
  return ((y * y * y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -3604.0, 23426.0), -63866.0), 93577.0), -77815.0), 34869.0), -6587.0)) * 0.001388888888888889);}


inline float alpha_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -3569.0, 16061.0), -27454.0), 21126.0), -6125.0), 49.0), -196.0), -36.0), 144.0)) * 2.48015873015873e-05);}
inline float beta_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 32121.0, -144548.0), 247074.0), -190092.0), 55125.0), -672.0), 2016.0), 512.0), -1536.0)) * 2.48015873015873e-05);}
inline float gamma_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -128484.0, 578188.0), -988256.0), 760312.0), -221060.0), 4732.0), -9464.0), -4032.0), 8064.0)) * 2.48015873015873e-05);}
inline float delta_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 299796.0, -1349096.0), 2305856.0), -1774136.0), 517580.0), -13664.0), 13664.0), 32256.0), -32256.0)) * 2.48015873015873e-05);}
inline float eta_l8_4(float y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, fma(y, fma(y, -449694.0, 2023630.0), -3458700.0), 2661540.0), -778806.0), 19110.0), -57400.0), 40320.0) * 2.48015873015873e-05);}
inline float zeta_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 449694.0, -2023616.0), 3458644.0), -2662016.0), 780430.0), -13664.0), -13664.0), 32256.0), 32256.0)) * 2.48015873015873e-05);}
inline float theta_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -299796.0, 1349068.0), -2305744.0), 1775032.0), -520660.0), 4732.0), 9464.0), -4032.0), -8064.0)) * 2.48015873015873e-05);}
inline float iota_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 128484.0, -578168.0), 988176.0), -760872.0), 223020.0), -672.0), -2016.0), 512.0), 1536.0)) * 2.48015873015873e-05);}
inline float kappa_l8_4(float y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -32121.0, 144541.0), -247046.0), 190246.0), -55685.0), 49.0), 196.0), -36.0), -144.0)) * 2.48015873015873e-05);}
inline float mu_l8_4(float y){
  return ((y * y * y * y * y * fma(y, fma(y, fma(y, fma(y, 3569.0, -16060.0), 27450.0), -21140.0), 6181.0)) * 2.48015873015873e-05);}
