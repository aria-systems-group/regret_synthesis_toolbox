import time
import warnings
import networkx as nx

from pathlib import Path
from networkx import DiGraph

from typing import List, Tuple

# A decorator to throw warning when we use deprecated methods/functions/routines
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


# A decorator to time the execution of a function
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


def get_nx_kosaraju_sort(game, debug: bool = False) -> Tuple[DiGraph, List[int]]:
    """
     A helper method tha Returns the condensation of graph G.

     The condensation of G is the graph with each of the strongly connected components
       contracted into a single node. Set Debug flag to true to print the SCC order
    """
    start = time.time()
    condensed_graph = nx.condensation(game._graph)
    stop = time.time()
    print(f"******************** Condensed Graph Computation: {stop - start} seconds ********************")
    
    scc_order = list(reversed(list(nx.topological_sort(condensed_graph))))
    if debug:
        print(f"SCC order in which the values iteration code shold run: {scc_order}")
        print(f"{max(scc_order) + 1} - Number of SCCs")

    return condensed_graph, scc_order


def is_docker():
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()