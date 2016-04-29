"""
@file remeshing_formula_parsing.py

Functions to parse some remeshing formula code (given as strings from Maple
or Sympy for instance). Result is a formula usable in the
hysop.numerics.remeshing module or in OpenCL code in
hysop/gpu/cl_src/remeshing/weights*
"""
import re
try:
    import sympy as sp

    # Weights names
    weights_names = ['alpha',
                     'beta',
                     'gamma',
                     'delta',
                     'eta',
                     'zeta',
                     'theta',
                     'iota',
                     'kappa',
                     'mu']

    def parse(f, fac=1, vec=False, toOpenCL=True,
              CLBuiltins=False, keep=False):
        """
        Parsing function.
        @param f : functions to parse as string
        @param fac : numeric factor for all formulas
        @param vec : is OpenCL output is generated to use vector builtin types
        @param toOpenCL : is OpenCL output
        @param CLBuiltins : is OpenCL output uses fma builtin function
        @param keep : low parsing
        """
        msg = 'Vector works only in OpenCL parsing'
        assert not (vec and not toOpenCL), msg
        assert not (CLBuiltins and not toOpenCL),\
            "CLBuiltins only in OpenCL parsing"
        t = "float__N__" if vec else "float"
        cteEnd = ".0" if toOpenCL else "."
        res = ""
        # Split each line
        fl = f.split('\n')
        # sympy formulas
        y = sp.symbols('y')
        print (y)
        sw = [None] * f.count(';')
        i = 0
        for wl in fl:
            if len(wl) > 2:
                # replace pow
                power = re.search('pow\(y, ([0-9]+)\)', wl)
                if power is not None:
                    np = "y" + "*y" * (int(power.group(1)) - 1)
                    wl = wl.replace(power.group(0), np)
                sw[i] = '('
                sw[i] += str(sp.horner(eval(wl.split(';')[0].split('=')[1]) * fac))
                sw[i] += ')/' + str(fac)
                i += 1
        for i, s in enumerate(sw):
            if not keep:
                if toOpenCL:
                    res += "inline " + t + " "
                    res += weights_names[i] + "(" + t + " y){\n"
                    res += '  return '
                else:
                    res += 'lambda y, s: s * '
                res += '('
                # replace y**n
                power = re.findall('y\*\*[0-9]+', s)
                if power is not None:
                    for pw in power:
                        n = int(pw.split('**')[1])
                        npower = 'y' + "*y" * (n - 1)
                        s = s.replace(pw, npower)
                s = s.replace(' ', '')
                if CLBuiltins:
                    s = createFMA(s)
                # From integers to floats
                s = re.sub(r"(?P<id>[0-9]+)", r"\g<id>" + cteEnd, s)
                s = s.replace('*', ' * ')
                s = s.replace('/', ' / ')
                s = s.replace('+', ' + ')
                s = s.replace('-', ' - ')
                s = s.replace('( - ', '(-')
                s = s.replace('  ', ' ')
                s = s.replace(", - ", ", -")
                res += s + ')'
                if toOpenCL:
                    res += ";}"
                if i < len(sw) - 1:
                    res += "\n"
            else:
                res += "w[{0}] = ".format(i)
                # replace y**n
                power = re.findall('y\*\*[0-9]+', s)
                if power is not None:
                    for pw in power:
                        n = int(pw.split('**')[1])
                        npower = 'y' + "*y" * (n - 1)
                        s = s.replace(pw, npower)
                # From integers to floats
                s = re.sub(r"(?P<id>[0-9]+)", r"\g<id>.", s)
                s = s.replace('*', ' * ')
                s = s.replace('/', ' / ')
                s = s.replace('+', ' + ')
                s = s.replace('-', ' - ')
                s = s.replace('( - ', '(-')
                s = s.replace('  ', ' ')
                s = s.replace(", - ", ", -")
                res += s + "\n"
        return res

except:
    msge = 'Sympy not available - remeshing formula parsing will not work.'
    msge += 'If you need parsing, try "pip install sympy" and reinstall hysop.'
    print msge


def createFMA(s):
    """
    Function to handle fma replacements in formula.
    @param s : formula to parse

    \code
    >>> createFMA("(y)")
    'fma(y, 1, 1)'
    >>> createFMA("(2*y)")
    'fma(y, 2, 1)'
    >>> createFMA("(y+11)")
    'fma(y, 1, 11)'
    >>> createFMA("(y+11)")
    'fma(y, 1, 11)'
    >>> createFMA("(y-11)")
    'fma(y, 1, -11)'
    >>> createFMA("(-y+11)")
    'fma(y, -1, 11)'
    >>> createFMA("(-y-11)")
    'fma(y, -1, -11)'
    >>> createFMA("(-22*y+11)")
    'fma(y, -22, 11)'
    >>> createFMA("(22*y-11)")
    'fma(y, 22, -11)'
    >>> createFMA("fma(y, 22, -11)")
    'fma(y, 22, -11)'
    >>> createFMA("(y*fma(y, 22, -11)+4)")
    'fma(y, fma(y, 22, -11), 4)'
    >>> createFMA("(y*fma(y, 22, -11)-4)")
    'fma(y, fma(y, 22, -11), -4)'
    >>> createFMA("(y*y*y*fma(y, 22, -11)+4)")
    'fma(y*y*y, fma(y, 22, -11), 4)'

    \endcode
    """
    def fma_replace(m):
        """
        Regexp callback function to replace a * y + c by fma(y, a, c).
        Matching regexp is (groups are given below):
        \code
        "\(((-)?(([0-9]+)\*)?)y(([+-])([0-9]+))?\)"
            |2|  |--4---|       |--6-||--7---|
                |-----3----|   |-------5------|
           |--------1--------|
        \endcode
        """
        s = "fma(y, "
        if not m.group(1) is None and not m.group(1) == '':
            # There is a '(-)?(a*)?'
            if not m.group(2) is None:
                # There is a '(-)' else '+' is not present and not needed
                s += m.group(2)
            if not m.group(3) is None:
                # There is a '(a*)' else 'a' is 1
                s += m.group(4)
            else:
                s += '1'
        else:
            s += '1'
        s += ', '
        if not m.group(5) is None:
            # There is a '(+-)(c)' else 'c' is 1
            if m.group(6) == '-':
                # There is a '-' else '+' is obmited
                s += m.group(6)
            s += m.group(7)
        else:
            s += '1'
        s += ')'
        return s

    def fma_recurse(m):
        """
        Regexp callback function to replace (y*fma(...)+c) by
        fma(y, fma(...), c) where '(' and ')' are well balanced.
        Matching regexp is (groups are given below):
        \code
        "\(([y\*]*y)\*(fma\(.*\))([+-])([0-9]+)\)"
           |---1---|  |----2----||-3--||--4---|
        \endcode
        """
        assert m.group(0).count('(') <= m.group(0).count(')'), \
            "Matching is too short to get ([fma(,,)]+): " + m.group(0)
        tmp = ""
        # get the part of the mathing that have the same number of '(' and ')'
        for t in m.group(0).split(')')[:m.group(0).count('(')]:
            tmp += t + ')'
        #performs the same regexp
        tmpg = re.search(r"\(([y\*]*y)\*(fma\(.*\))([+-])([0-9]+)\)", tmp)
        s = "fma(" + tmpg.group(1) + ", " + tmpg.group(2) + ', '
        if tmpg.group(3) == '-':
            s += tmpg.group(3)
        s += tmpg.group(4)
        s += ')'
        return m.group(0).replace(tmp, s)

    s = re.sub(r"\(((-)?(([0-9]+)\*)?)y(([+-])([0-9]+))?\)", fma_replace, s)
    l = len(re.findall(r'fma\(', s))
    recurse = True
    while recurse:
        s = re.sub(r"\(([y\*]*y)\*(fma\(.*\))([+-])([0-9]+)\)", fma_recurse, s)
        ll = len(re.findall(r'fma\(', s))
        if l == ll:
            recurse = False
        l = ll
    return s
