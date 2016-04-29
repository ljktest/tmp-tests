"""
@file profiling.py

External tool to profile python scripts
This scripts performs a time and memory profiling.
The time profiling is obtained by cProfile module and memory by meliae module.
Only a 'run' function is monitored.
"""
import argparse
import cProfile
import pstats
import os
import string
import sys
from meliae import scanner
from meliae import loader

# Argument parser
parser = argparse.ArgumentParser(description='Profile a given pyton script.')
parser.add_argument('-pyscript', required=True,
                   help='a python script to profile.')
parser.add_argument('-prefix', default='./',
                   help='output files prefix.')
parser.add_argument('-comm', default=" ",
                   help='Program description.')

# get arguments values
args = vars(parser.parse_args())
# Import all in given script
sys.path.append('.')
exec ("from " + args['pyscript'] + " import *")

rawMemoryOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_result.meliae'
MemorySizeOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_result.meliae_size'
rawCProfileOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_result.pstat'
cProfileTimeOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_result.pstat_time'
cProfileCumulativeOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_result.pstat_cumulative'
callTreePngOutput = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_calltree.png'
profileSummaryTex = args['prefix']+args['pyscript'].replace('.','_') +\
    '_profile_summary.tex'
outputDir = callTreePngOutput.rsplit('/',1)[0]
callTreePngOutputFileName = callTreePngOutput.rsplit('/',1)[1]
profileSummaryTexFileName = profileSummaryTex.rsplit('/',1)[1]


def create_mem_stat(f):
    """create Memory stats from file
    @param f file to load
    """
    om = loader.load(f)
    om.summarize()
    thesummary = loader._ObjSummary()
    objs = om.objs.itervalues()
    for obj in objs:
        thesummary._add(obj)
    if thesummary.summaries is None:
        thesummary.by_size()
    return thesummary

def get_mem_stats(thesummary, amount=None):
    """print stats function for pstats datas
    @param thesummary : meliae summary to display
    @param amount : number of items to display
    """
    if amount == None:
        amount = len(thesummary.summaries)
    cumulative = 0
    for i in xrange(amount):
        summary = thesummary.summaries[i]
        cumulative += summary.total_size
        yield (i, summary.count,
               summary.count * 100.0 / thesummary.total_count,
               summary.total_size,
               summary.total_size * 100.0 / thesummary.total_size,
               cumulative * 100.0 / thesummary.total_size, summary.max_size,
               summary.type_str)

def print_mem_stat_file(memRawStats, f):
    """
    Prints the given summary in file f
    @param memRawStats : Meliae profiling output
    @param f : file to write
    """
    f = open(f, 'w')
    f.write("\n   Total {0} objects, ".format(memRawStats.total_count))
    f.write("{0} types, ".format(len(memRawStats.summaries)))
    f.write("Total size = {0:.1f}MiB ({1} bytes)\n\n".format(
        memRawStats.total_size / 1024. / 1024, memRawStats.total_size))
    f.write(" Index    Count      %     Size      %    Cum      Max Kind \n")
    for elem in get_mem_stats(memRawStats):
        f.write("{0:6d} {1:8d} {2:6.2f} {3:8d} ".format(
                elem[0], elem[1], elem[2], elem[3]))
        f.write("{0:6.2f} {1:6.2f} {2:8d} {3:s}\n".format(
                elem[4], elem[5], elem[6], elem[7]))
    f.write("\n\n")
    f.close()

def get_stats(stats, amount):
    """print stats function for pstats datas
    @param stats : profiling data
    @param amount : number of element to display
    """
    if stats.fcn_list:
        lst = stats.fcn_list[:]
        msg = "   Ordered by: " + stats.sort_type + '\n'
    else:
        lst = stats.stats.keys()
        msg = "   Random listing order was used\n"
    for selection in xrange(amount):
        lst, msg = stats.eval_print_amount(selection, lst, msg)
    if list:
        for func in lst:
            cc, nc, tt, ct, callers = stats.stats[func]
            yield (cc, tt, ct, "%s:%d(%s)" % func)


def f8(x):
    return "%8.3f" % x


def print_line_file(stats, func, f):
    """
    Print a line of the profiling output for the given function in file f
    @param stats : profiling data
    @param func : function to print
    @param f : file to write
    """
    cc, nc, tt, ct, callers = stats.stats[func]
    c = str(nc)
    if nc != cc:
        c = c + '/' + str(cc)
    f.write(str(c.rjust(9)) + ' ')
    f.write(str(f8(tt)) + ' ')
    if nc == 0:
        f.write(str(' ' * 8) + ' ')
    else:
        f.write(str(f8(tt / nc)) + ' ')
    f.write(str(f8(ct)) + ' ')
    if cc == 0:
        f.write(str(' ' * 8) + ' ')
    else:
        f.write(str(f8(ct / cc)) + ' ')
    f.write("%s:%d(%s)" % func)
    f.write("\n")


def print_stats_file(stats, f, amount):
    """
    Print profiling data in file.
    @param stats : profiling data
    @param f : file to write
    @param amount : number of items to display
    """
    f = open(f, 'w')
    for ff in stats.files:
        f.write(str(ff))
    if stats.files:
        f.write("\n")
        indent = ' ' * 8
        for func in stats.top_level.keys():
            f.write(str(indent) + ' ' + str(func_get_function_name(func)))

        f.write(str(indent) + ' ' + str(stats.total_calls) + ' ' + str("function calls"))
        if stats.total_calls != stats.prim_calls:
            f.write(str("(%d primitive calls)" % stats.prim_calls))
        f.write(str("in %.3f CPU seconds" % stats.total_tt))
        f.write("\n")
        width = stats.max_name_len
        if stats.fcn_list:
            ls = stats.fcn_list[:]
            msg = "   Ordered by: " + stats.sort_type + '\n'
        else:
            ls = stats.stats.keys()
            msg = "   Random listing order was used\n"

        for selection in xrange(amount):
            ls, msg = stats.eval_print_amount(selection, ls, msg)

        count = len(ls)

        if not ls:
            return 0, ls
        f.write(msg)
        if count < len(stats.stats):
            width = 0
            for func in ls:
                if len(func_std_string(func)) > width:
                    width = len(func_std_string(func))
        if ls:
            f.write('   ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n')
            for func in ls:
                print_line_file(stats, func, f)
        f.write("\n")
        f.write("\n")
        f.close()


def launch_meliae_profiling():
    """Lanch of a run function first to get memory profiling with meliae"""
    run()

    scanner.dump_all_objects(rawMemoryOutput)
    memRawStats = create_mem_stat(rawMemoryOutput)
    print_mem_stat_file(memRawStats, f=MemorySizeOutput)
    stat = "Total {0} objects, {1} types, Total size = {2:.1f}MiB " + \
        "({3} bytes)\n".format(memRawStats.total_count,
                               len(memRawStats.summaries),
                               memRawStats.total_size / 1024. / 1024,
                               memRawStats.total_size)
    stat += " Index    Count      %     Size      %    Cum      Max Kind \n"
    statPercent = 0
    statPieChart = ""
    i = 0
    for l in get_mem_stats(memRawStats, 10):
        stat += "{0:6d} {1:8d} {2:6.2f} {3:8d} {4:6.2f} {5:6.2f} {6:8d} " + \
            "{7:s}\n".format(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])
        if i < 10:
            statPieChart += "{0:6.2f}/ {1:s} ".format(
                l[4], string.replace("\\texttt{"+l[7]+"}",'_','\_'))
            if i != 9:
                statPieChart += ","
            statPercent += l[4]
            i += 1
            #statPieChart += "{0:6.2f}/ {1:s}".format(100-statPercent, "\\texttt{Others}")
    return stat, statPieChart


def launch_cProfile():
    cProfile.run('run()', rawCProfileOutput)

    prof = pstats.Stats(rawCProfileOutput)
    #p.strip_dirs().sort_stats('time').print_stats()
    print_stats_file(prof.strip_dirs().sort_stats('time'),
                     cProfileTimeOutput, 15)
    print_stats_file(prof.strip_dirs().sort_stats('cumulative'),
                     cProfileCumulativeOutput, 15)

    # Calling gprof2dot to make graphic output calltree
    os.system("gprof2dot -f pstats " + rawCProfileOutput + \
                  " | dot -Tpng -o "+callTreePngOutput)
    return prof


p = launch_cProfile()
memStat, memStatPieChart = launch_meliae_profiling()
# Create results string
dataFiles = ""
for filename in p.files:
    dataFiles += filename + ' '
cpuTotalTime = 'Total CPU time : ' + str(p.total_tt)
pstatTime = "  ncalls  tottime      %   cumtime     % " + \
    "filename:lineno(function) \n"
pstatTimePercent = 0
pstatTimeStringPieChart = ""
ii = 0
for l in get_stats(p.strip_dirs().sort_stats('time'), 15):
    pstatTime+="{0:8d} {1:8.4f} {2:6.2f} {3:8.4f} {4:6.2f} " + \
        "{5:s}\n".format(l[0], l[1], 100*l[1]/p.total_tt, l[2],
                         100*l[2]/p.total_tt, l[3])
    if ii < 10:
        pstatTimeStringPieChart += "{0:6.2f}/ {1:s} ".format(
            100*l[1]/p.total_tt, string.replace("\\texttt{"+l[3]+"}",'_','\_'))
        if ii != 9:
            pstatTimeStringPieChart += ","
        pstatTimePercent += 100*l[1]/p.total_tt
        ii += 1
        #pstatTimeStringPieChart += "{0:6.2f}/ {1:s}".format(100-pstatTimePercent, "\\texttt{Others}")

pstatCum = "  ncalls  tottime      %   cumtime     % " + \
    "filename:lineno(function) \n"
for l in get_stats(p.strip_dirs().sort_stats('cumulative'), 15):
    pstatCum+="{0:8d} {1:8.4f} {2:6.2f} {3:8.4f} {4:6.2f} {5:s}\n".format(
        l[0], l[1], 100*l[1]/p.total_tt, l[2], 100*l[2]/p.total_tt, l[3])


f_out = open(profileSummaryTex, 'w')
f_out.write("""\\documentclass[11pt]{scrartcl}
\\usepackage[top=1cm,bottom=1cm,left=1cm,right=1cm]{geometry}
\\usepackage{float,caption}
\\usepackage{pgf-pie}
\\usepackage{lscape}
\\usepackage{multicol}

\\begin{document}
\\begin{landscape}
\\begin{multicols}{2}
\\begin{verbatim}
"""
+ dataFiles
+ """
\\end{verbatim}

"""
+ args['comm']
+ """\\\\

"""
+ cpuTotalTime
+ """
\\section*{Top 15 CPU total time}

\\begin{footnotesize}
\\begin{verbatim}
"""
+ pstatTime
+ """
\\end{verbatim}
\\end{footnotesize}


\\section*{Top 15 CPU cumulative time}

\\begin{footnotesize}
\\begin{verbatim}
"""
+ pstatCum
+ """
\\end{verbatim}
\\end{footnotesize}

\\columnbreak
\\section*{Top 10 used memory}

\\begin{footnotesize}
\\begin{verbatim}
"""
+ memStat
+ """
\\end{verbatim}
\\end{footnotesize}

\\begin{figure}[H]
  %\\centering
\\begin{tikzpicture}[scale=0.75,font=\\footnotesize]
\\pie[explode=0,text=legend]{""" + pstatTimeStringPieChart + """}
\\end{tikzpicture}
  \\caption{Top 5 CPU total time}
\\end{figure}


\\begin{figure}[H]
  %\\centering
\\begin{tikzpicture}[scale=0.75,font=\\footnotesize]
\\pie[explode=0,text=legend]{""" + memStatPieChart + """}
\\end{tikzpicture}
  \\caption{Top 5 Used memory}
\\end{figure}

\\end{multicols}
\\end{landscape}
\\newpage
\\section*{Call graph}
\\begin{figure}[H]
  \\centering
  \\includegraphics[width=\\textwidth]{""" + callTreePngOutputFileName + """}
  \\caption{Call graph}
\\end{figure}


\\end{document}""")
f_out.close()

# Compiling LaTeX summary
cmd = "cd " + outputDir + ";"
cmd += "pdflatex -interaction batchmode " + profileSummaryTexFileName + ";"
cmd += "rm " + profileSummaryTexFileName.replace(".tex", ".aux") + ";"
cmd += "rm " + profileSummaryTexFileName.replace(".tex", ".log") + ";"
cmd += "cd -"
os.system(cmd)
