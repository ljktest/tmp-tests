m4p = """
w[0] = (-1 + (2 - y) * y) * y / 2;
w[1] = 1 + (-5 + 3 * y) * y * y / 2;
w[2] = (1 + (4 - 3 * y) * y) * y / 2;
w[3] = (y - 1) * y * y / 2;
"""

m4p_python = """lambda y, s: s * ((y * (y * (-y + 2.) - 1.)) / 2.)
lambda y, s: s * ((y * y * (3. * y - 5.) + 2.) / 2.)
lambda y, s: s * ((y * (y * (-3. * y + 4.) + 1.)) / 2.)
lambda y, s: s * ((y * y * (y - 1.)) / 2.)"""

m4p_cl_novec = """inline float alpha(float y){
  return ((y * (y * (-y + 2.0) - 1.0)) / 2.0);}
inline float beta(float y){
  return ((y * y * (3.0 * y - 5.0) + 2.0) / 2.0);}
inline float gamma(float y){
  return ((y * (y * (-3.0 * y + 4.0) + 1.0)) / 2.0);}
inline float delta(float y){
  return ((y * y * (y - 1.0)) / 2.0);}"""

m4p_cl = """inline float__N__ alpha(float__N__ y){
  return ((y * (y * (-y + 2.0) - 1.0)) / 2.0);}
inline float__N__ beta(float__N__ y){
  return ((y * y * (3.0 * y - 5.0) + 2.0) / 2.0);}
inline float__N__ gamma(float__N__ y){
  return ((y * (y * (-3.0 * y + 4.0) + 1.0)) / 2.0);}
inline float__N__ delta(float__N__ y){
  return ((y * y * (y - 1.0)) / 2.0);}"""

m4p_cl_builtin = """inline float__N__ alpha(float__N__ y){
  return ((y * fma(y, fma(y, -1.0, 2.0), -1.0)) / 2.0);}
inline float__N__ beta(float__N__ y){
  return (fma(y * y, fma(y, 3.0, -5.0), 2.0) / 2.0);}
inline float__N__ gamma(float__N__ y){
  return ((y * fma(y, fma(y, -3.0, 4.0), 1.0)) / 2.0);}
inline float__N__ delta(float__N__ y){
  return ((y * y * fma(y, 1.0, -1.0)) / 2.0);}"""

l6_cl_builtin = """inline float__N__ alpha(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -89.0, 312.0),"""
l6_cl_builtin += """ -370.0), 140.0), 15.0), 4.0), -12.0)) / 720.0);}
inline float__N__ beta(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 623.0, -2183.0)"""
l6_cl_builtin += """, 2581.0), -955.0), -120.0), -54.0), 108.0)) / 720.0);}
inline float__N__ gamma(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -1869.0, 6546.0),"""
l6_cl_builtin += """ -7722.0), 2850.0), 195.0), 540.0), -540.0)) / 720.0);}
inline float__N__ delta(float__N__ y){
  return (fma(y * y, fma(y * y, fma(y, fma(y, fma(y, 3115.0, -10905.0),"""
l6_cl_builtin += """ 12845.0), -4795.0), -980.0), 720.0) / 720.0);}
inline float__N__ eta(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -3115.0, 10900.0),"""
l6_cl_builtin += """ -12830.0), 4880.0), -195.0), 540.0), 540.0)) / 720.0);}
inline float__N__ zeta(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, 1869.0, -6537.0),"""
l6_cl_builtin += """ 7695.0), -2985.0), 120.0), -54.0), -108.0)) / 720.0);}
inline float__N__ theta(float__N__ y){
  return ((y * fma(y, fma(y, fma(y, fma(y, fma(y, fma(y, -623.0, 2178.0),"""
l6_cl_builtin += """ -2566.0), 1010.0), -15.0), 4.0), 12.0)) / 720.0);}
inline float__N__ iota(float__N__ y){
  return ((y * y * y * y * fma(y, fma(y, fma(y, 89.0, -311.0), 367.0),"""
l6_cl_builtin += """ -145.0)) / 720.0);}"""

l63 = """w[0] = (-12 + (4 + (15 + (140 + (-370 + (312 - 89 * y) * y) * y)"""
l63 += """* y) * y) * y) * y / 720;
w[1] = (108 + (-54 + (-120 + (-955 + (2581 + (-2183 + 623 * y) * y) * y)"""
l63 += """* y) * y) * y) * y / 720;
w[2] = (-540 + (540 + (195 + (2850 + (-7722 + (6546 - 1869 * y) * y) * y)"""
l63 += """ * y) * y) * y) * y / 720;
w[3] = 1 + (-980 + (-4795 + (12845 + (-10905 + 3115 * y) * y) * y) * y * y)"""
l63 += """* y * y / 720;
w[4] = (540 + (540 + (-195 + (4880 + (-12830 + (10900 - 3115 * y) * y) * y)"""
l63 += """* y) * y) * y) * y / 720;
w[5] = (-108 + (-54 + (120 + (-2985 + (7695 + (-6537 + 1869 * y) * y) * y)"""
l63 += """* y) * y) * y) * y / 720;
w[6] = (12 + (4 + (-15 + (1010 + (-2566 + (2178 - 623 * y) * y) * y) * y)"""
l63 += """* y) * y) * y / 720;
w[7] = (-145 + (367 + (-311 + 89 * y) * y) * y) * pow(y, 4) / 720;
"""

try:
    from hysop.tools.remeshing_formula_parsing import parse

    def test_parsing_toPython():
        assert m4p_python == parse(m4p, 2, toOpenCL=False)

    def test_parsing_toOpenCL():
        assert m4p_cl_novec == parse(m4p, 2, vec=False, toOpenCL=True)
        assert m4p_cl == parse(m4p, 2, vec=True, toOpenCL=True)
        assert m4p_cl_builtin == \
            parse(m4p, 2, vec=True, toOpenCL=True, CLBuiltins=True)
        res = parse(l63, 720, vec=True, toOpenCL=True, CLBuiltins=True)
        assert l6_cl_builtin == res

except:
    pass
