from pydot import Dot, Node, Edge

graph = Dot(graph_type='digraph')

graph.add_edge(Edge('a','b'))

graph.write_png('test.png')