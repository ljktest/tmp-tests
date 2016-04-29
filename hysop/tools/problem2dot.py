"""
@file problem2dot.py
Converts a problem instance to a graph throw dot syntax.

"""
from hysop.operator.advection import Advection
from hysop.operator.redistribute import Redistribute
from hysop.operator.redistribute_inter import RedistributeInter
from hysop.mpi.main_var import main_rank
import pydot
colors = [
    "#dc322f",
    "#859900",
    "#268bd2",
    "#b58900",
    "#d33682",
    "#2aa198",
    "#cb4b16",
    "#6c71c4",
    "#ffffff"]


def get_shape(op):
    if isinstance(op, Redistribute) or isinstance(op, RedistributeInter):
        return 'octagon'
    else:
        return 'box'


def toDot(pb, filename='graph.pdf'):
    if main_rank == 0:
        all_ops = []
        all_vars = []
        tasks = []
        for op in pb.operators:
            if isinstance(op, Advection) and not op.advecDir is None:
                for ad_op in op.advecDir:
                    all_ops.append(ad_op)
                    tasks.append(ad_op.task_id)
            else:
                all_ops.append(op)
                tasks.append(op.task_id)
            for v in op.variables:
                all_vars.append(v)
        all_vars = list(set(all_vars))
        tasks = list(set(tasks))

        all_edges = {}
        for op_id, op in enumerate(all_ops):
            for v in op.input:
                out = None
                for req in op.wait_list():
                    if v in req.output:
                        out = req
                i = op_id-1
                while(out is None):
                    if v in all_ops[i].output:
                        if isinstance(all_ops[i], RedistributeInter):
                            if op == all_ops[i].opTo:
                                out = all_ops[i]
                        else:
                            if not isinstance(all_ops[i], Redistribute):
                                out = all_ops[i]
                    i = i-1
                if (out, op) in all_edges.keys():
                    all_edges[(out, op)].append(v.name)
                else:
                    all_edges[(out, op)] = [v.name]

        graph = pydot.Dot(pb.__class__.__name__, graph_type='digraph')
        sub_graphs = {}
        nodes = {}
        edges = {}
        from_start = {}
        to_end = {}
        # Start iteration node
        G_start = pydot.Node(-1, label="START", shape='none')
        graph.add_node(G_start)
        # End iteration node
        G_end = pydot.Node(-2, label="END", shape='none')
        graph.add_node(G_end)
        if len(tasks) > 1:
            for t in tasks:
                if t is None:
                    c = 'white'
                else:
                    c = colors[tasks.index(t)]
                sub_graphs[t] = pydot.Subgraph('cluster_' + str(t),
                                               label='',
                                               color=c)
        else:
            sub_graphs[tasks[0]] = graph
        for op_id, op in enumerate(all_ops):
            label = 'Op' + str(op_id) + '_' + op.name
            nodes[op] = pydot.Node(op_id, label=label,
                                   shape=get_shape(op))
            sub_graphs[op.task_id].add_node(nodes[op])
        for e in all_edges.keys():
            if all_ops.index(e[0]) < all_ops.index(e[1]):
                edges[e] = pydot.Edge(nodes[e[0]], nodes[e[1]],
                                      label='_'.join(list(set(all_edges[e]))),
                                      color='black')
                graph.add_edge(edges[e])
            else:
                if (e[0], G_end) in to_end.keys():
                    to_end[(e[0], G_end)] += all_edges[e]
                else:
                    to_end[(e[0], G_end)] = all_edges[e]
                if (G_start, e[1]) in from_start.keys():
                    from_start[(G_start, e[1])] += all_edges[e]
                else:
                    from_start[(G_start, e[1])] = all_edges[e]
        for e in to_end.keys():
            edges[e] = pydot.Edge(nodes[e[0]], e[1],
                                  label='_'.join(list(set(to_end[e]))),
                                  color='black')
            graph.add_edge(edges[e])
        for e in from_start.keys():
            edges[e] = pydot.Edge(e[0], nodes[e[1]],
                                  label='_'.join(list(set(from_start[e]))),
                                  color='black')
            graph.add_edge(edges[e])
        if len(tasks) > 1:
            for t in tasks:
                graph.add_subgraph(sub_graphs[t])
        graph.write(filename + '.dot', format='dot')
        graph.write(filename, format=filename.split('.')[-1])
