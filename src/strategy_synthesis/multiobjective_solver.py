import copy
import multiprocessing
import queue
import time
from abc import ABCMeta
from typing import Any, Dict, Hashable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

from src.graph import TwoPlayerGraph
from src.prism.strategy import DeterministicStrategy, StochasticStrategy
from src.strategy_synthesis.adversarial_game import ReachabilityGame

Strategy = Any
INFINITY = 1e10
Node = Hashable
Nodes = List[Node]
EdgeWeight = List[float]
Point = List[float]
ParetoPoint = List[float]
ParetoPoints = List[ParetoPoint]
Strategies = Dict[str, Strategy]

NUM_CORES = multiprocessing.cpu_count()


def get_vertices(polygon: Polygon) -> List:
    if not isinstance(polygon, Polygon):
        the_polygon = None
        for p in polygon:
            if isinstance(p, Polygon):
                the_polygon = p
        polygon = the_polygon

    X, Y = polygon.exterior.xy

    vertices = set()
    for x, y in zip(X, Y):
        vertices.add((x, y))
    return list(vertices)


class ParetoFront(metaclass=ABCMeta):
    """Polyhedron"""

    _polyhedron = None

    def __init__(
        self,
        player: str,
        points: Union[int, float, List, np.ndarray] = None,
        dim: int = None,
        distance: int = np.inf,
        convex: bool = False,
        upperbound: float = INFINITY,
        minimization: bool = True,
        name: str = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ):
        """
        Pareto Front Class at each node.
        It keeps either a list of pareto points or a polyhedron.
        The polyhedron could be convex or non-convex.

        :args points:           1D or 2D array
        :args convex:           Whether the polygon would be convex or not
        :args upperbound:       The upperbound of the polyhedron

        e.g.,
            array = [[1], [2]]
            array = [[1,2], [3,4]]
            array = [[1,2,3], [4,5,6]]

        If 1D array is given, it will be converted as 2D
        e.g.,
            array = [1, 2] -> [[1], [2]]
        """
        if player not in ["eve", "adam"]:
            raise Exception("Select either player = eve / adam")

        self._player = player
        self._upperbound = INFINITY if upperbound == 0 else upperbound
        self._distance = distance
        self._convex = convex
        self._minimization = (
            minimization  # TODO: In the future, adapt the code to maximization
        )
        self._name = name
        self._rtol = rtol
        self._atol = atol

        points = self._check_valid_points(points)

        if points is not None:
            self._dim = len(points[0])
        elif dim is not None:
            self._dim = dim
        else:
            raise Exception("Provide either points or dimension of the polyhedron")

        if points is not None:
            self._polyhedron = self._construct_polyhedron(
                points, self._upperbound, self._convex
            )

    def _check_valid_points(self, points: Union[List, np.ndarray]):
        """
        Check if the given points have the correct type and dimension

        :arg points:    A list of points. It must be either a 1D or 2D array
        :return:        2D Array
        """
        # 1. Check if valid array
        if points is None or len(points) == 0:
            return None

        if isinstance(points, (int, float)):
            points = [[points]]

        if not isinstance(points, (List, np.ndarray)):
            raise Exception("points must be a list or np.ndarray")

        # 2. Check for the dimensions of the array
        def dim(_array):
            """ "return shape of a multi-dimensional array (1st, 2nd, ...)"""
            return (
                [len(_array)] + dim(_array[0])
                if (isinstance(_array, (List, np.ndarray)))
                else []
            )

        ndim = len(dim(points))

        # Only accept 1D or 2D
        if ndim >= 3:
            raise ValueError("points must be 1D or 2D array")

        # If 1D, make it 2D
        if ndim == 1:
            if isinstance(points, List):
                points = [points]
            else:
                points = np.array([points])

        elem_size = len(points[0])

        # Check if the size of the point is same as the other points
        for point in points:
            if point is None:
                return None

            if not isinstance(point, (List, np.ndarray)):
                raise Exception("A point must be a list or np.ndarray")

            if len(point) != elem_size:
                msg = f"A point of a set must be a size of {elem_size}"
                raise ValueError(msg)

        return points

    def _construct_polyhedron(
        self, points: Union[List, np.ndarray], upperbound: float, convex: bool
    ):
        if convex:
            return self._construct_convex_polyhedron(points, upperbound)
        else:
            return self._construct_nonconvex_polyhedron(points, upperbound)

    def _construct_nonconvex_polyhedron(
        self, points: Union[List, np.ndarray], upperbound: float
    ):
        """
        Construct a non-convex polygon from points of the polygon
        TODO: Currently, limited to polygons (2D Polyhedron) and
        dimensions other than 2D are not supported.
        """
        polygons = []
        for p in points:
            polygon = Polygon(
                [p, (p[0], upperbound), (upperbound, upperbound), (upperbound, p[1])]
            )
            polygons.append(polygon)

        if len(polygons) == 1:
            return polygons[0]

        polygon_union = polygons[0]
        for polygon in polygons[1:]:
            polygon_union = polygon_union.union(polygon)

        return polygon_union

    def _construct_convex_polyhedron(
        self, points: Union[List, np.ndarray], upperbound: float
    ):
        """
        Construct a convex polygon from points of the polygon
        TODO: Currently, limited to polygons (2D Polyhedron) and
        dimensions other than 2D are not supported.
        """
        # Sort by the first column
        columnIndex = 0
        points = np.array(points)
        points = points[points[:, columnIndex].argsort()].tolist()

        points = (
            [[points[0][0], upperbound]]
            + points
            + [[upperbound, points[-1][1]], [upperbound, upperbound]]
        )

        return Polygon(points)

    def get_polyhedron_vertices(
        self, polyhedron=None, exclude_threshold: bool = True, epsilon: int = 5
    ):
        if polyhedron is None:
            if not self.is_initialized():
                return None
            polyhedron = self._polyhedron

        vertices = get_vertices(polyhedron)

        vertices_wo_threshold = []
        for vertex in vertices:
            vertex = list(vertex)
            if self._upperbound not in vertex:
                # if any(np.around(vertex, epsilon) == self._upperbound):
                vertices_wo_threshold.append(vertex)

        return self._get_minimum_element_set(vertices_wo_threshold)

    def _get_minimum_element_set(self, points: Union[List, np.ndarray]):
        points = np.array(points)
        elem_size = len(points[0])

        delete_indices = []
        for i, current in enumerate(points):
            points_wo_current = copy.deepcopy(points)
            points_wo_current[i] = [-1] * elem_size

            for i, vertex in enumerate(points_wo_current):
                if all(current <= vertex):
                    # delete the element
                    delete_indices.append(i)

        new_points = np.delete(points, delete_indices, axis=0)

        return new_points

    def update(self, successor_pareto_fronts: List):
        """
        In this function, our goal is to find the minimum maximum pareto points
        that the system player can guarantee to achieve.
        In other word, while the environment player tried to play adversarial
        (maximum weights), the system player needs to find a value set that
        can be reached at any successor node.
        """
        if self._player == "eve":
            self._polyhedron = self._union_of_successor_fronts(successor_pareto_fronts)
            # self._polyhedron = self._union_of_successor_fronts(successor_pareto_fronts + [self])
            self._distance = min([s._distance for s in successor_pareto_fronts]) + 1
        else:
            self._polyhedron = self._intersection_of_successor_fronts(
                successor_pareto_fronts
            )
            self._distance = max([s._distance for s in successor_pareto_fronts]) + 1

    def _union_of_successor_fronts(self, successor_pareto_fronts: List):
        successor_pareto_fronts_ = []
        for front in successor_pareto_fronts:
            if front.is_initialized():
                successor_pareto_fronts_.append(front)

        if len(successor_pareto_fronts_) == 0:
            return None

        if len(successor_pareto_fronts_) == 1:
            return successor_pareto_fronts_[0]._polyhedron

        # # Take intersections of all those polyhedra
        # polyhedron = successor_pareto_fronts_[0]._polyhedron
        # for front in successor_pareto_fronts_[1:]:
        #     if front.is_initialized():
        #         polyhedron = polyhedron.union(front._polyhedron)

        polyhedra = [f._polyhedron for f in successor_pareto_fronts_]
        polyhedron = cascaded_union(polyhedra)

        return polyhedron

    def _intersection_of_successor_fronts(self, successor_pareto_fronts: List):
        if len(successor_pareto_fronts) == 0:
            return None

        for front in successor_pareto_fronts:
            if not front.is_initialized():
                return None

        if len(successor_pareto_fronts) == 1:
            return successor_pareto_fronts[0]._polyhedron

        # Take intersections of all those polyhedrons
        polyhedron = successor_pareto_fronts[0]._polyhedron
        for front in successor_pareto_fronts[1:]:
            polyhedron = polyhedron.intersection(front._polyhedron)

        return polyhedron

    def get_intersection_point_with(self, point):
        if len(point) != 2:
            raise Exception("We only allow a list of 2 elements")

        polygon = Polygon([(0, 0), (point[0], 0), point, (0, point[1])])

        # Intersection
        polygon = polygon.intersection(self._polyhedron)
        points = self.get_polyhedron_vertices(polygon)

        return points

    # TODO: Assume pareto points could be infinities
    def expand_by(self, weights: List[float]) -> None:
        if not self.is_initialized():
            return

        # 1. get the pareto points.
        pareto_points = self.pareto_points

        # Get the dimension of the polyhedron
        ndim = len(pareto_points[0])

        # Check if the given weights matches with the dimension of the polyhedron
        if len(weights) != ndim:
            raise Exception(f"The dimension of {weights} must be {ndim}")

        # 2. Expanded the pareto points by weights
        pareto_points = [
            (np.array(pareto_point) + np.array(weights)).tolist()
            for pareto_point in pareto_points
        ]

        # 3. Construct the polyhedron from the expanded points
        self._polyhedron = self._construct_polyhedron(
            pareto_points, self._upperbound, self._convex
        )

    def __add__(self, weights: Union[List, np.ndarray]) -> "ParetoFront":
        if not isinstance(weights, (List, np.ndarray)):
            raise Exception('Must be in a form of "ParetoFront + List/np.ndarray"')

        if isinstance(weights, np.ndarray):
            weights = np.array(weights)

        updated_pareto_front = copy.deepcopy(self)
        updated_pareto_front.expand_by(weights)

        return updated_pareto_front

    def __str__(self):
        return str(self.pareto_points)

    def is_initialized(self):
        return self._polyhedron is not None

    def __eq__(self, others):
        if np.array(self.pareto_points).shape != np.array(others.pareto_points).shape:
            return False
        # return np.array_equal(self.pareto_points, others.pareto_points)
        return np.allclose(
            self.pareto_points, others.pareto_points, self._rtol, self._atol
        )

    @property
    def pareto_points(self) -> List[List[float]]:
        if not self.is_initialized():
            if self._minimization:
                return np.inf * np.ones((1, self._dim))
            else:
                return np.zeros((1, self._dim))

        try:
            pareto_points = self.get_polyhedron_vertices()
        except:
            print(self._polyhedron)
            msg = "Polyhedron is not a Polyhedron!"
            msg += "I don't know why, but there is something wrong with Shapely"
            raise Exception(msg)

        return pareto_points


class MultiObjectiveSolver:
    """
    Currently, only supported for two weights, weight and auto_weight.
    TODO: Support for multiple objective functions

    :attribute _Wg_graph: Graph with Game Weights
    :attribute _Wa_graph: Graph with Automaton Weights
    """

    def __init__(
        self,
        game: TwoPlayerGraph = None,
        stochastic: bool = False,
        adversarial: bool = False,
        epsilon: float = 1e-8,
        round_decimals: int = 10,
        round_label_decimals: int = 2,
        max_iteration: int = 20,
    ):
        """ """
        self._game = game

        self._stochastic = stochastic
        self._adversarial = adversarial

        if -np.log10(epsilon) > round_decimals:
            raise Exception("epsilon has to be larger than round_decimals")

        self._epsilon = epsilon
        self._round_decimals = round_decimals
        self._round_label_decimals = round_label_decimals
        self._max_iteration = max_iteration
        self._previously_seen = set()
        self._notconverged_state = None
        self._count = 1

        self._initialize(game)

    def _initialize(self, game: TwoPlayerGraph):
        if game is None:
            return

        self._game = copy.deepcopy(game)

        self._reachability_solver = ReachabilityGame(game=self._game)
        self._reachability_solver.reachability_solver()

        init_node = self._game.get_initial_states()[0][0]
        if init_node not in self._reachability_solver.sys_winning_region:
            raise Exception(f"{init_node} not in the system's winning region")
        self._game.delete_selfloops()

        self._upperbounds = self._compute_upperbound_for_pareto_points()

        self._pareto_fronts = None
        self._strategies = None

    def is_initialized(self):
        return self._game is not None

    def _compute_upperbound_for_pareto_points(self) -> List[float]:
        """
        Compute the maximum possible values that can be achieved in the game
        """
        max_depth = len(self._game._graph.nodes())

        max_weights = []
        for name in self._game.weight_types:
            weights = [
                e[2]["weights"].get(name)
                for e in self._game._graph.edges.data()
                if "weights" in e[2]
            ]
            print(
                f"Weights: {weights}, Type: {name}, Graph Edge: {len(self._game._graph.edges)}"
            )
            # TODO: (kandai) Fix this shit
            max_single_weight = max(weights)
            max_weights.append(max_depth * max_single_weight)

        return max_weights

    def get_strategies(self):
        if self._strategies is None:
            raise Exception("No strategies found. Please run 'solve' or 'solve' first.")

        return self._strategies

    def get_a_strategy_for(self, pareto_point: ParetoPoint):
        strategies = self.get_strategies()

        if not isinstance(pareto_point, Tuple):
            pareto_point = tuple(pareto_point)

        pareto_point = self._get_closest_pareto_point(pareto_point)

        if pareto_point is None:
            raise Exception(f"Could not find {pareto_point} in the pareto points")

        if pareto_point not in strategies:
            raise Exception(f"{pareto_point} is not in the computed pareto points")

        return strategies[pareto_point]

    def _round_pareto_point(self, pareto_point: ParetoPoints) -> ParetoPoints:
        return tuple(np.around(pareto_point, self._round_decimals))

    def _round_pareto_points(self, pareto_points: ParetoPoints) -> ParetoPoints:
        pareto_points = np.around(pareto_points, self._round_decimals)
        return list(map(tuple, pareto_points))

    def _get_closest_pareto_point(self, pareto_point: ParetoPoint) -> ParetoPoint:
        pareto_points = np.array(self.get_pareto_points())
        idx = np.linalg.norm(pareto_points - pareto_point, axis=1).argmin()

        nearest_pareto_point = pareto_points[idx]
        zeros = np.zeros(len(pareto_point))
        rounded_pareto_point = self._round_pareto_point(
            nearest_pareto_point - pareto_point
        )

        if any(rounded_pareto_point != zeros):
            return None
        else:
            return tuple(nearest_pareto_point)

    def get_pareto_points(self, round_value: bool = False):
        if self._pareto_fronts is None:
            raise Exception(
                "No pareto points found. Please run 'solve' function first."
            )

        init_node = self._game.get_initial_states()[0][0]
        pareto_points = self._pareto_fronts[init_node].pareto_points

        if round_value:
            return self._round_pareto_points(pareto_points)

        return pareto_points

    def get_pareto_point_at(self, node, round_value: bool = False):
        if self._pareto_fronts is None:
            raise Exception(
                "No pareto points found. Please run 'solve' function first."
            )

        if node not in self._game._graph.nodes():
            raise Exception(f"{node} is not in the game graph")

        pareto_points = self._pareto_fronts[node].pareto_points

        if round_value:
            return self._round_pareto_points(pareto_points)

        return pareto_points

    def is_pareto_computed(self, pareto_fronts=None) -> bool:
        if pareto_fronts is None:
            pareto_fronts = self._pareto_fronts
        return pareto_fronts is not None

    def check_if_pareto_computed(self, pareto_fronts=None) -> None:
        if not self.is_pareto_computed(pareto_fronts):
            msg = "No pareto fronts found. Please run 'solve' or 'solve_pareto_points' first."
            raise Exception(msg)

    def solve(
        self,
        game: TwoPlayerGraph = None,
        bound: Point = None,
        plot_pareto: bool = False,
        plot_all_paretos: bool = False,
        plot_strategies: bool = False,
        plot_graph_with_pareto: bool = False,
        plot_graph_with_strategy: bool = False,
        speedup: bool = True,
        debug: bool = False,
        view: bool = False,
        format: str = "svg",
    ) -> Tuple[ParetoPoints, Strategies]:
        pareto_points = self.solve_pareto_points(
            game,
            plot_pareto,
            plot_all_paretos,
            plot_graph_with_pareto,
            speedup,
            debug,
            view,
            format,
        )

        strategies = self.solve_strategies(
            bound, plot_strategies, plot_graph_with_strategy, debug, view, format
        )

        return pareto_points, strategies

    def solve_pareto_points(
        self,
        game: TwoPlayerGraph = None,
        plot_pareto: bool = True,
        plot_all_paretos: bool = False,
        plot_graph_with_pareto: bool = False,
        speedup: bool = True,
        debug: bool = False,
        view: bool = False,
        format: str = "svg",
    ) -> ParetoPoints:
        if not self.is_initialized():
            if game is None:
                raise Exception('Initialize first by providing a "game" graph')
            else:
                self._initialize(game)

        # Pareto Computation
        start = time.time()
        self._pareto_fronts = self._compute_pareto_points(speedup, debug)
        end = time.time()
        print(f"Pareto Points Computation took {end-start:.2f} seconds")

        # Pareto Visualization
        if plot_pareto:
            self.plot_pareto_front()

        if plot_all_paretos:
            self.plot_pareto_fronts()

        self._label_pareto_on_graph()

        if plot_graph_with_pareto:
            if self._notconverged_state is not None:
                self.plot_partial_graph(
                    self._notconverged_state, view=view, format=format
                )
            else:
                self.plot_game(view=view, format=format)

        return self.get_pareto_points()

    def solve_strategies(
        self,
        bound: Point = None,
        plot_strategies: bool = False,
        plot_graph_with_strategy: bool = False,
        debug: bool = False,
        view: bool = False,
        format: str = "svg",
    ) -> Strategies:
        pareto_points = self.get_pareto_points()

        if bound is not None:
            pareto_points = self._compute_points_in_pareto_front(bound, pareto_points)

        strategies = {}
        for pareto_point in pareto_points:
            strategy = self._compute_strategy(pareto_point, debug)
            strategies[tuple(pareto_point)] = strategy

            if plot_strategies:
                self.plot_strategy(strategy, view, format)

            if plot_graph_with_strategy:
                self.plot_graph_with_strategy(self._game, strategy, view, format)

        self._strategies = strategies

        return self.get_strategies()

    def _compute_points_in_pareto_front(
        self, bound: Point, pareto_points: ParetoPoints
    ) -> List[Point]:
        """
        Given a constraint and a pareto front at the initial state,
        compute a point in the pareto front
        """
        # If the user chooses a point that is completely inside the pareto front
        # Just return the pareto points
        points = []
        for pareto_point in pareto_points:
            if all(np.array(bound) >= np.array(pareto_point)):
                points.append(pareto_point.tolist())

        if len(points) != 0:
            return points

        # Otherwise, find the intersection
        init_node = self._game.get_initial_states()[0][0]

        return self._pareto_fronts[init_node].get_intersection_point_with(bound)

    # Pareto Points Computation
    def _compute_pareto_points(self, speedup: bool = True, debug: bool = True):
        stochastic = self._stochastic
        adversarial = self._adversarial
        game = self._game

        init_node = game.get_initial_states()[0][0]
        accept_node = game.get_accepting_states()[0]
        next_node = list(game._graph.successors(init_node))[0]

        weights = game.get_edge_attributes(init_node, next_node, "weights")
        n_weight = len(weights)

        pareto_fronts = {}
        for node in game._graph.nodes():
            player = game.get_state_w_attribute(node, "player")

            if player == "adam" and not adversarial:  # System / Robot
                pareto_fronts[node] = ParetoFront(
                    "eve", name=node, dim=n_weight, convex=stochastic
                )
            else:
                pareto_fronts[node] = ParetoFront(
                    player, name=node, dim=n_weight, convex=stochastic
                )

        pareto_fronts[accept_node] = ParetoFront(
            "eve", np.zeros(n_weight), name=accept_node, distance=0, convex=stochastic
        )
        pareto_fronts_prev = None

        visitation_order = self._decide_visitation_order()
        iter_var = 0

        if debug:
            print(
                f"{len(game._graph.nodes())} nodes and {len(game._graph.edges())} edges"
            )

        while not self._converged(pareto_fronts, pareto_fronts_prev):
            if debug:
                # print(f"{iter_var} Iterations")
                start = time.time()

            if iter_var > self._max_iteration:
                self._print_diff(pareto_fronts, pareto_fronts_prev)
                msg = "Pareto Front Computation did not converge."
                msg += "\nTry setting epsilon 100 larger than round_decimals,"
                msg += "\nOr increase max_iteration to allow more iterations."
                raise Exception(msg)

            pareto_fronts_prev = copy.deepcopy(pareto_fronts)
            iter_var += 1

            for u_node in visitation_order:
                # Only allow transitions to the winning region
                # This prevents from pareto points monotonically increasing in the losing region.
                # This is due to the fact that Shapely cannot handle infinity values.
                if u_node not in self._reachability_solver.sys_winning_region:
                    continue

                player = game.get_state_w_attribute(u_node, "player")

                # Get Successors' Pareto Fronts + Edge Weights
                fronts = []
                for v_node in game._graph.successors(u_node):
                    edge_weight = list(
                        game.get_edge_attributes(u_node, v_node, "weights").values()
                    )

                    if speedup:
                        fronts.append(pareto_fronts[v_node] + edge_weight)
                    else:
                        fronts.append(pareto_fronts_prev[v_node] + edge_weight)

                # Update u_node's Pareto Front by either taking the union / intersection
                pareto_fronts[u_node].update(fronts)

            if debug:
                end = time.time()
                print(f"{iter_var}th Iteration took {end-start:.2f} seconds")

        return pareto_fronts

    def _decide_visitation_order(self):
        accept_node = self._game.get_accepting_states()[0]

        search_queue = queue.Queue()
        visited = set()
        visitation_order = []

        search_queue.put(accept_node)
        visited.add(accept_node)
        visitation_order.append(accept_node)

        while not search_queue.empty():
            v_node = search_queue.get()

            for u_node in self._game._graph.predecessors(v_node):
                # Avoid self loop
                if u_node == v_node:
                    continue

                if u_node not in visited:
                    visitation_order.append(u_node)
                    visited.add(u_node)
                    search_queue.put(u_node)

        visitation_order.remove(accept_node)

        return visitation_order

    def _converged(self, curr: Dict, prev: Dict) -> bool:
        if curr is None or prev is None:
            return False

        curr_keys = list(curr.keys())
        prev_keys = list(prev.keys())

        if curr_keys != prev_keys:
            return False

        for key in curr_keys:
            if curr[key] != prev[key]:
                curr_pp = tuple(map(tuple, curr[key].pareto_points.tolist()))
                prev_pp = tuple(map(tuple, prev[key].pareto_points.tolist()))

                if (key, curr_pp) in self._previously_seen and (
                    key,
                    prev_pp,
                ) in self._previously_seen:
                    if self._count == 1:
                        self._notconverged_state = key
                        return True
                    self._count += 1

                self._previously_seen.add((key, curr_pp))

                return False

        return True

    def _print_diff(self, curr: Dict, prev: Dict) -> None:
        if curr is None or prev is None:
            return
        curr_keys = list(curr.keys())
        prev_keys = list(prev.keys())
        if curr_keys != prev_keys:
            print(curr_keys, prev_keys)
        for key in curr_keys:
            if curr[key] != prev[key]:
                print(key, curr[key], prev[key])

    # Strategy Computation
    def _compute_strategy(
        self, init_pareto_point: ParetoPoint, debug: bool = False
    ) -> Strategy:
        # Strategy Synthesis
        if self._stochastic:
            strategy = StochasticStrategy.from_pareto_fronts(
                self._game,
                self._pareto_fronts,
                epsilon=self._round_decimals,
                init_pareto_point=init_pareto_point,
            )
        else:
            strategy = DeterministicStrategy.from_pareto_fronts(
                self._game,
                self._pareto_fronts,
                epsilon=self._round_decimals,
                init_pareto_point=init_pareto_point,
                adversarial=self._adversarial,
            )

        return strategy

    # Plots
    def _label_pareto_on_graph(
        self,
        game: TwoPlayerGraph = None,
        pareto_fronts=None,
        round_label_decimals: int = None,
    ) -> None:
        if game is None:
            game = self._game
        else:
            game = copy.deepcopy(game)

        if pareto_fronts is None:
            pareto_fronts = self._pareto_fronts

        if round_label_decimals is None:
            round_label_decimals = self._round_label_decimals

        self.check_if_pareto_computed(pareto_fronts)

        node_labels = {}
        for node in game._graph.nodes():
            pareto_points = pareto_fronts[node].pareto_points

            node_labels[node] = np.array2string(
                np.array(pareto_points).round(round_label_decimals)
            )
        game.set_node_labels_on_fancy_graph(node_labels)

        return game

    def plot_game(
        self, view: bool = True, format: str = "png", add_label: bool = False
    ):
        if add_label:
            self._label_pareto_on_graph()
        self._game.plot_graph(view=view, format=format)

    def plot_partial_graph(
        self, start_state, view, format, n_immediate_children: int = 3
    ):
        init_node = self._game.get_initial_states()[0][0]
        accept_node = self._game.get_accepting_states()[0]

        search_queue = queue.Queue()
        search_queue.put((0, start_state))
        nodes_to_keep = [start_state, init_node, accept_node]
        edges_to_keep = []

        while not search_queue.empty():
            ith, u_node = search_queue.get()

            if ith == n_immediate_children:
                continue

            for v_node in self._game._graph.successors(u_node):
                if v_node not in nodes_to_keep:
                    nodes_to_keep.append(v_node)
                    edges_to_keep.append((u_node, v_node))
                    search_queue.put((ith + 1, v_node))

        # Construct a game graph with these nodes
        game = copy.deepcopy(self._game)
        edges = copy.deepcopy(list(game._graph.edges()))
        for edge in edges:
            if edge[0] not in nodes_to_keep and edge[0] not in nodes_to_keep:
                game._graph.remove_edge(*edge)

        nodes = copy.deepcopy(list(game._graph.nodes()))
        for node in nodes:
            if node not in nodes_to_keep:
                game._graph.remove_node(node)

        game.plot_graph(view=view, format=format)

    def plot_pareto_front(self, ax=None) -> None:
        self.check_if_pareto_computed()

        init_node = self._game.get_initial_states()[0][0]
        next_node = list(self._game._graph.successors(init_node))[0]

        weights = self._game.get_edge_attributes(init_node, next_node, "weights")
        weight_names = list(weights.keys())

        points = self._pareto_fronts[init_node].pareto_points
        points = np.array(points)
        points = points[points[:, 1].argsort()]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.scatter(points[:, 0], points[:, 1], marker="o")

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # for x, y in zip(points[:, 0], points[:, 1]):
        #     ax.vlines(x, ymin=y, ymax=ymax)
        #     ax.hlines(y, xmin=x, xmax=xmax)

        n_point = points.shape[0]
        for i in range(n_point):
            x = points[i, 0]
            y = points[i, 1]
            xmax_ = xmax if i == 0 else points[i - 1, 0]
            ymax_ = ymax if i == n_point - 1 else points[i + 1, 1]
            ax.vlines(x, ymin=y, ymax=ymax_)
            ax.hlines(y, xmin=x, xmax=xmax_)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.set_xlabel(weight_names[0])
        ax.set_ylabel(weight_names[1])

        plt.show()

    def plot_pareto_fronts(self, include_successors: bool = True) -> None:
        self.check_if_pareto_computed()
        pass

    def plot_graph_with_strategy(
        self,
        game: TwoPlayerGraph,
        strategy: Strategy,
        view: bool = False,
        format: str = "svg",
    ) -> None:
        game = copy.deepcopy(game)
        edges = strategy.get_edges_on_original_graph()
        game.set_strategy(edges)
        game.graph_name += "With" + strategy._graph_name
        game.plot_graph(view=view, format=format)

    def plot_strategy(self, strategy: Strategy, view: bool = True, format: str = "png"):
        strategy.plot_graph(view=view, format=format)

    def plot_strategies(self, view: bool = True, format: str = "png"):
        for pp in self.get_pareto_points():
            strategy = self.get_a_strategy_for(pp)
            self.plot_strategy(strategy, view, format)
