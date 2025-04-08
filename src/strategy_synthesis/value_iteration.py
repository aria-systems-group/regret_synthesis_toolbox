import sys
import math
import copy
import operator
import warnings
import numpy as np

from bidict import bidict
from collections import defaultdict
from typing import Optional, Union, Dict, List, Tuple

from numpy import ndarray

# import local packages
from ..graph import TwoPlayerGraph
from ..helper_methods import deprecated
from .adversarial_game import ReachabilityGame as ReachabilitySolver

# numpy int32 min value
INT_MIN_VAL = -2147483648
INT_MAX_VAL = 2147483647


class ValueIteration:

    def __init__(self, game: TwoPlayerGraph, competitive: bool = False, int_val: bool = True,  prior_val_vector: Optional[dict] = None):
        self.org_graph: Optional[TwoPlayerGraph] = copy.deepcopy(game)
        self.competitive = competitive
        self._int_val: bool = int_val
        self._val_vector: Optional[ndarray] = None
        self._node_int_map: Optional[bidict] = None
        self._num_of_nodes: int = 0
        self._state_value_dict: Optional[dict] = defaultdict(lambda: -1)
        self._str_dict: Dict = defaultdict(lambda: -1)
        self._sys_str_dict: Dict = defaultdict(lambda: -1)
        self._env_str_dict: Dict = defaultdict(lambda: -1)
        self._sys_winning_region: set = None
        self._winning_region: set = None
        self._accp_states: set = set(self.org_graph.get_accepting_states())
        self._iterations_to_converge = math.inf
        self._convergence_dict = defaultdict(lambda: -1)
        self._prior_val_vector: dict = prior_val_vector
        self._init_state = self.set_init_state()
        if prior_val_vector:
            self._initialize_warm_started_val_vector()
        else:
            self._initialize_val_vector()
        

    @property
    def org_graph(self):
        return self.__org_graph
    
    @property
    def init_state(self):
        return self._init_state

    @property
    def val_vector(self):
        return self._val_vector

    @property
    def competitive(self):
        return self._competitive

    @property
    def num_of_nodes(self):
        return self._num_of_nodes

    @property
    def state_value_dict(self):
        return self._state_value_dict

    @property
    def str_dict(self):
        return self._str_dict

    @property
    def sys_str_dict(self):
        return self._sys_str_dict

    @property
    def env_str_dict(self):
        return self._env_str_dict

    @property
    def node_int_map(self):
        return self._node_int_map
    
    @property
    def sys_winning_region(self):
        return self._sys_winning_region
    
    @property
    def winning_region(self):
        return self._winning_region
    

    @property
    def prior_val_vector(self):
        return self._prior_val_vector
    
    @property
    def iterations_to_converge(self):
        if self._iterations_to_converge == math.inf:
            warnings.warn("[Error] Please Run the Value Iteration's solve() method before accessing the `iterations_to_converge` attribute.")
            sys.exit(-1)
        return self._iterations_to_converge
    
    @property
    def convergence_dict(self):
        if self._iterations_to_converge == math.inf:
            warnings.warn("[Error] Please Run the Value Iteration's solve() method before accessing the `convergence_dict` attribute.")
            sys.exit(-1)
        
        self._convergence_dict = self._compute_convergence_idx()
        return self._convergence_dict

    @org_graph.setter
    def org_graph(self, org_graph):
        if len(org_graph._graph.nodes()) == 0:
            warnings.warn("Please make sure that the graph is not empty")
            sys.exit(-1)

        if not isinstance(org_graph, TwoPlayerGraph):
            warnings.warn("The Graph should be of type of TwoPlayerGraph")
            sys.exit(-1)

        self.__org_graph = org_graph

    @competitive.setter
    def competitive(self, value: bool):
        self._competitive = value


    def set_init_state(self):
        try:
            _init_state: List[tuple] = self.org_graph.get_initial_states()
            assert len(_init_state) == 1, "The initial state should be a single."
            return _init_state[0][0]
        except:
            return None
        
    

    def is_winning(self) -> bool:
        """
        A helper method that return True if the initial state(s) belongs to the list of system player' winning region
        :return: boolean value indicating if system player can force a visit to the accepting states or not
        """
        if INT_MIN_VAL < self.state_value_dict.get(self.init_state) < INT_MAX_VAL:
            return True
        return False

    def _initialize_target_state_costs(self):
        """
        A method that computes the set of target states, and assigns the respective nodes a zero value in the the value
        vector

        For non-absorbing games (products that are constrcuted in non-abdorbing fashion), we need to make sure that the
          env states with accepting DFA state should NOT initlaized as target state. This is because, the env player controls
          that state. Thus we will only initialize the sys states with accepting DFA state as target state with initial value 0. 
        :return:
        """
        assert len(self._accp_states) != 0, "For Value Iteration algorithm, you need atleast one accepting state. FIX THIS!!!"

        for _s in self._accp_states:
            if self.org_graph.get_state_w_attribute(_s, "player") == "eve":
                _node_int = self.node_int_map[_s]
                self.val_vector[_node_int][0] = 0

    def _initialize_trap_state_costs(self):
        """
        A method that computes the set of trap states, and assigns the respective node a zero value in the value vector.
        The weights transiting to trap state however are given a big finite value in this case. This is done to make
        sure the values associated with states from where you cannot reach the accepting state(s) (The trap region in an
        adversarial game) have a finite (but big) values rather than all having the states in the trap region having inf
        values
        :return:
        """

        _trap_states = self.org_graph.get_trap_states()

        # if there is no trap state or nothing transits to the trap state then initialize the accepting state as the
        # target state
        if len(_trap_states) == 0:
            warnings.warn("Trap state not found: Initializing cooperative game with accepting state as the target"
                          " vertex")
            self._initialize_target_state_costs()

        for _trap_state in _trap_states:
            if len(list(self.org_graph._graph.predecessors(_trap_state))) == 1:
                warnings.warn("Trap state not found: Initializing cooperative game with accepting state as the target"
                              " vertex")
                self._initialize_target_state_costs()

        for _s in _trap_states:
            _node_int = self.node_int_map[_s]
            self.val_vector[_node_int][0] = 0

    def _initialize_val_vector(self):
        """
        A method to initialize parameters such the
         1. Number of nodes in a given graph,
         2. The internal node mapping dict, and
         3. Initializing the value vector and assigning all the target state(s) 0 init value
        :return:
        """
        self._num_of_nodes = len(list(self.org_graph._graph.nodes))
        self._node_int_map = bidict({state: index for index, state in enumerate(self.org_graph._graph.nodes)})
        self._val_vector = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)
        self._initialize_target_state_costs()
    

    def _initialize_warm_started_val_vector(self):
        """
        A method to initialize parameters such the
        :return:
        """
        self._num_of_nodes = len(list(self.org_graph._graph.nodes))
        self._node_int_map = bidict({state: index for index, state in enumerate(self.org_graph._graph.nodes)})
        self._val_vector = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)
        try:
            for state, state_val in self.prior_val_vector.items():
                self._val_vector[self._node_int_map[state]] = state_val
        except KeyError:
            pass

    def _is_same(self, pre_val_vec, curr_val_vec):
        """
        A method to check is if the two value vectors are same or not
        :param pre_val_vec:
        :param curr_val_vec:
        :return:
        """
        return np.array_equal(pre_val_vec, curr_val_vec)

    def _add_trap_state_player(self):
        """
        A method to add a player to the trap state, if any
        :return:
        """

        _trap_states = self.org_graph.get_trap_states()

        for _n in _trap_states:
            self.org_graph.add_state_attribute(_n, "player", "adam")
    
    @deprecated
    def cooperative_solver(self, debug: bool = False, plot: bool = False):
        """
        A Method to compute the cooperative value from each state when both players are playing minimally.

        All states except for the target states (accepting states) are initialized to at infinity. We start from the accepting state and propagate back our costs.
        While doing so, both players are acting the same, minimally, essentially turning this game into a single
        player game. The algorithm terminates when we reach a fixed point.

        THe initial state will have finite state value.

        :param debug: A Flag to print the iteration number we are at
        :param plot: A flag to plot the strategies as well as the values that states converged to
        :return:
        """

        # initially in the org val_vector the target node(s) will value 0
        _init_node = self.org_graph.get_initial_states()[0][0]

        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        iter_var: int = 0

        _str_dict = {}

        while not self._is_same(_val_pre, _val_vector):
            if debug:
                if iter_var % 1000 == 0:
                    print(f"{iter_var} Iterations")

            _val_pre = copy.copy(_val_vector)
            iter_var += 1

            for _n in self.org_graph._graph.nodes():
                # are we making an assumption that there is only one accepting state?
                if _n in self._accp_states:
                    continue

                _int_node = self.node_int_map[_n]

                # we don't need to check if that state belong to adam or eve. They both behave the same.
                _val_vector[_int_node][0], _next_min_node = self._get_min_sys_val(_n, _val_pre)
                if _val_vector[_int_node] != _val_pre[_int_node]:
                    _str_dict[_n] = self.node_int_map.inverse[_next_min_node]

            self._val_vector = np.append(self.val_vector, _val_vector, axis=1)

        # safely convert values in the last col of val vector to ints
        if self._int_val:
            _int_val_vector = self.val_vector[:, -1].astype(int)
        else:
            _int_val_vector = self.val_vector[:, -1]

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            if _int_val_vector[i] < 0:
                self.state_value_dict.update({_s: math.inf})
            else:
                self.state_value_dict.update({_s: _int_val_vector[i]})

        self._str_dict = _str_dict

        if plot:
            self._change_orig_graph_name(prefix='coop_str_on_')
            self._add_state_costs_to_graph()
            self.add_str_flag()
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            self._sanity_check()

    def _change_orig_graph_name(self, prefix: str = None,
                                      suffix: str = None,
                                      name: str = None):
        if all(arg is None for arg in [prefix, suffix, name]):
            raise ValueError('Please provide at least one argument')

        if name:
            graph_name = name

        graph_name = self.__org_graph._graph.name

        if prefix:
            graph_name = prefix + graph_name

        if suffix:
            graph_name = graph_name + suffix

        self.__org_graph._graph.name = graph_name
    
    def _get_opt_val(self, node: Union[str, tuple], pre_vec: ndarray):
        """
        This method only return the max (adverasrial)/min (cooperative) value
        """
        _succ_vals = set({})
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = (self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int][0])
            _succ_vals.add(_val)
        
        if self.org_graph.get_state_w_attribute(node, "player") == "eve":
            _val = min(_succ_vals)
        else:
            #TODO: Fix this in future
            # assert self.org_graph.get_state_w_attribute(node, "player") == "adam", "Error. Encountered a state that is supposed to belong to the env player."
            if self.competitive:
                _val = max(_succ_vals)
            else:
                _val = min(_succ_vals)

        return _val


    def extract_strategy(self) -> Tuple[dict, dict]:
        """
         A method that extracts the optimal strategy to take at each state add it to the Sys and Env dictionary

         At Sys state: argmin_a(F(s, a) + W(s')) for all s'  
         At Env state: argmax_a(F(s, a) + W(s')) for all s' 
        
        Here F(s, a) is the edge weight and W(s') is the value of the successor state in previous iteration.
        We assume that the Sys player is trying to minimize the total cost it expends. 
        """
        _env_str_dict = {}
        _sys_str_dict = {}
        for _n in self.org_graph._graph.nodes():
            # get the max value
            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                _next_max_node = self._get_max_env_val(_n, self.val_vector[:, -1])
                if _next_max_node is not None:
                    _env_str_dict[_n] = _next_max_node
            
            # get the min value
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                _next_min_node = self._get_min_sys_val(_n, self.val_vector[:, -1])
                if _next_min_node is not None:
                    _sys_str_dict[_n] = _next_min_node 
        
        return _sys_str_dict, _env_str_dict
    
    def update_state_values(self, val_vector: ndarray) -> ndarray:
        """
        A method that back propagates the state values. Specfically, at the

            At Sys state: min_a(F(s, a) + W(s')) for all s'  
            At Env state: max_a(F(s, a) + W(s')) for all s' 

        Here F(s, a) is the edge weight and W(s') is the value of the successor state in previous iteration.
        We assume that the Sys player is trying to minimize the total cost it expends. 
        """

        val_pre = copy.copy(val_vector)

        for _n in self.org_graph._graph.nodes():
            if self.prior_val_vector is not None and _n in self.prior_val_vector.keys():
                continue
            
            _int_node = self.node_int_map[_n]

            if _n in self._accp_states and self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                continue
            
            val_vector[_int_node][0] = self._get_opt_val(_n, val_pre)

        self._val_vector = np.append(self.val_vector, val_vector, axis=1)

        return val_vector

    def solve(self, debug: bool = False, plot: bool = False, extract_strategy: bool = True):
        """
        A method that implements Algorithm 1 from the paper. The operation performed at each step can be represented by
        an operator say F.  F here is the _get_max_env_val() and _get_min_sys_val() methods. F is a monotonic operator
        and is monotonically decreasing - meaning the function should not increase (it must not increase!) and converges
        to the greatest fixed point of F.

        As all the weights are positive in our case, the state values monotonically decrease and converge to the greatest
        fixed point.

        The Val of the game is infact the Greatest Fixed Point.  The upper bound on the # of iterations to converge is
        (2|V| -1)W|V| + |V|.
        :param debug:
        :param plot:
        :return:
        """
        # initially in the org val_vector the target node(s) will value 0
        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        iter_var = 0

        while not self._is_same(_val_pre, _val_vector):
            if debug:
                # if iter_var % 1000 == 0:
                print(f"{iter_var} Iterations")

            _val_pre = copy.copy(_val_vector)
            iter_var += 1

            # perform one step Value Iteration
            _val_vector: ndarray = self.update_state_values(val_vector=_val_vector)
        
        self._iterations_to_converge = iter_var

        # safely convert values in the last col of val vector to ints
        if self._int_val:
            _int_val_vector = self.val_vector[:, -1].astype(int)
        else:
            _int_val_vector = self.val_vector[:, -1]

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            # the above conversion converts math.inf to negative vals, we restore them to be inf
            if _int_val_vector[i] < 0:
                self.state_value_dict.update({_s: math.inf})
            else:
                self.state_value_dict.update({_s: _int_val_vector[i]})
        
        # extract sys and env strategy after converging.
        if extract_strategy:
            self._sys_str_dict, self._env_str_dict = self.extract_strategy()

        self._str_dict = {**self._sys_str_dict, **self._env_str_dict}
        self._sys_winning_region = set(self._sys_str_dict.keys()) #.union(self._accp_states)
        self._winning_region = set([s for s, val in self.state_value_dict.items() if val != math.inf])

        if plot:
            self._change_orig_graph_name(prefix='adv_str_on_')
            self._add_state_costs_to_graph()
            self.add_str_flag()
            self.org_graph.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            _init_node = self.org_graph.get_initial_states()[0][0]
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            self._sanity_check()
            # self.print_state_values()

    def _sanity_check(self):
        """
        A nice charateristic of the algorithm is that the vertices from which you cannot reach the target set (assuming
        human to be purely adversarial) is exactly the region W2(the trap region) from the adversarial game solver. So
        this method checks if this property holds after the state values have computed
        :return:
        """
        adv_solver = ReachabilitySolver(self.org_graph)
        adv_solver.reachability_solver()
        _trap_region = set(adv_solver.env_winning_region)

        _num_of_states, _ = self.val_vector.shape

        _cumu_trap_region = set()
        for _i in range(_num_of_states):
            _node = self.node_int_map.inverse[_i]
            # if this state has INT_MAX_VAL as it final value then check it belong to the trap region
            if self.val_vector[_i][-1] == INT_MAX_VAL:
                _cumu_trap_region.add(_node)

        if _cumu_trap_region == _trap_region:
            print("The two sets are equal")

    def _compute_convergence_idx(self) -> Dict[int, int]:
        """
        This method is used to determine when each state in the graph converged to their values. A state value is
        converged if x_{k} != x_{k-1} where k is the kth iteration and x is a state in the graph
        :return:
        """
        _convergence_dict: dict = {}
        _num_of_states, _num_of_iter = self.val_vector.shape

        for _state in range(_num_of_states):
            _converge_at_first_iter = True
            for _itr in range(_num_of_iter - 1, 0, -1):
                if self.val_vector[_state][_itr] != self.val_vector[_state][_itr - 1]:
                    _converge_at_first_iter = False
                    _convergence_dict.update({self.node_int_map.inverse[_state]: _itr})
                    break
            if _converge_at_first_iter:
                _convergence_dict.update({self.node_int_map.inverse[_state] : 0})

        return _convergence_dict
    

    def _get_state_sorted_by_convergence(self) -> Dict[tuple, int]:
        """
        This method is used to looks thorugh the convergence diectionary and sorts states according to the iteration at which they converged.
         This method is useful in debugging. 
        :return:
        """
        _states_by_convergence_itr_dict: dict = defaultdict(lambda : set())

        for state, itr in self.convergence_dict.items():
            _states_by_convergence_itr_dict[itr].add(state)

        return _states_by_convergence_itr_dict

    def print_state_values(self):
        """
        A method to print the state value
        :return:
        """

        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            print(f"State {_s} Value {self.val_vector[i]}")

    def _get_max_env_val(self, node: Union[str, tuple], pre_vec: ndarray) -> str:
        """
        A method that returns the max value for the current node that belongs to the env.
        :param node: The current node in the graph
        :param pre_vec: The previous value vector
        :return: The optimal state to transition to.
        """

        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            _succ_vals.append((_next_n, _val))

        # get org node int value
        if self.competitive:
            _next_node, _ = max(_succ_vals, key=operator.itemgetter(1))
        else:
            _next_node, _val = min(_succ_vals, key=operator.itemgetter(1))
            if _val == math.inf:
                return None
        return _next_node
        

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray) -> Union[str, None]:
        """
        A method that returns the min value of the current node that belongs to the sys
        :param node: The current node in the graph
        :param pre_vec: The previous value vector
        :return: The optimal state to transition to if successor states values are not Inf.
        """

        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            if _val != math.inf:
                _succ_vals.append((_next_n, _val))

        try:
            _next_node, _ = min(_succ_vals, key=operator.itemgetter(1))
            return _next_node
        except ValueError: 
            return None


    def _add_state_costs_to_graph(self):
        """
        A helper method that computes the costs associated with each state to reach the accepting state and add it to
        the nodes.
        :return:
        """
        for _n in self.org_graph._graph.nodes():
            self.org_graph.add_state_attribute(_n, "val", [self.state_value_dict[_n]])

    def add_str_flag(self):
        """

        :param str_dict:
        :return:
        """
        self.org_graph.set_edge_attribute('strategy', False)

        for curr_node, next_node in self._str_dict.items():
            if isinstance(next_node, list):
                for n_node in next_node:
                    self.org_graph._graph.edges[curr_node, n_node, 0]['strategy'] = True
            else:
                self.org_graph._graph.edges[curr_node, next_node, 0]['strategy'] = True


class PermissiveValueIteration(ValueIteration):
    """
    Inherit Value Iteration class and override the max, min function to return a set of strategies.

    The solve() function is modified to store set (permissive) of optimal strategies. 
    """

    def __init__(self, game: TwoPlayerGraph, competitive: bool = False, int_val: bool = True, prior_val_vector: Optional[dict] = None):
        super().__init__(game, competitive, int_val, prior_val_vector)
    

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray) -> Union[List[str], None]:
        """
        A method that returns the min value of the current node that belongs to the sys
        :param node: The current node in in the graph
        :param pre_vec: The previous value vector
        :return: A List of states
        """
        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = pre_vec[_node_int] + self.org_graph.get_edge_weight(node, _next_n)
            if _val != math.inf:
                _succ_vals.append((_next_n, _val))

        try:
            _, min_val = min(_succ_vals, key=operator.itemgetter(1))
            return [_node for _node, _node_val in _succ_vals if min_val == _node_val]
        except ValueError:
            return None

    
    def _get_max_env_val(self, node: Union[str, tuple], pre_vec: ndarray) -> List[str]:
        """
        A method that returns the max value for the current node that belongs to the env.
        :param node: The current node in in the graph
        :param pre_vec: The previous value vector
        :return: A List of states
        """

        _succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = pre_vec[_node_int] + self.org_graph.get_edge_weight(node, _next_n)
            _succ_vals.append((_next_n, _val))

        # get org node int value
        if self.competitive:
            _, _val = max(_succ_vals, key=operator.itemgetter(1))
        else:
            _, _val = min(_succ_vals, key=operator.itemgetter(1))
            if _val == math.inf:
                return None

        return [_node for _node, _node_val in _succ_vals if _val == _node_val]

    def plot_graph(self):
        """
         A helper function that changes the originla name of the graph, add state cost as attribute, adds strategy
           flags to strategies and
        """
        self._change_orig_graph_name(prefix='adv_str_on_')
        self._add_state_costs_to_graph()
        self.add_str_flag()
        self.org_graph.plot_graph()
    
    def solve(self, debug: bool = False, plot: bool = False, extract_strategy: bool = True):
        """
         TODO: Looks like there isn't much difference in implementation of the parent method's solve(). Can I get rid of this function?
            :param debug:
            :param plot:
            :return:
        """
        # initially in the org val_vector the target node(s) will value 0
        _init_node = self.org_graph.get_initial_states()[0][0]

        _val_vector = copy.deepcopy(self.val_vector)
        _val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        iter_var = 0

        while not self._is_same(_val_pre, _val_vector):
            if debug:
                # if iter_var % 1000 == 0:
                print(f"{iter_var} Iterations")

            _val_pre = copy.copy(_val_vector)
            iter_var += 1

            # perform one step Value Iteration
            _val_vector: ndarray = self.update_state_values(val_vector=_val_vector)

        # safely convert values in the last col of val vector to ints
        if self._int_val:
            _int_val_vector = self.val_vector[:, -1].astype(int)
        else:
            _int_val_vector = self.val_vector[:, -1]

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: _int_val_vector[i] if INT_MIN_VAL <  _int_val_vector[i] < INT_MAX_VAL  else math.inf})

        # extract sys and env strategy after converging.
        if extract_strategy:
            self._sys_str_dict, self._env_str_dict = self.extract_strategy()

        self._str_dict = {**self._sys_str_dict, **self._env_str_dict}
        self._sys_winning_region = set(self._sys_str_dict.keys()) #.union(self._accp_states)
        self._winning_region = set([s for s, val in self.state_value_dict.items() if val != math.inf])

        if plot:
            self.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            print(f"Init state value: {self.state_value_dict[_init_node]}")
            # self._sanity_check()

    def _add_state_costs_to_graph(self):
        """
        A helper method that computes the costs associated with each state to reach the accepting state and add it to
        the nodes.
        :return:
        """

        for _n in self.org_graph._graph.nodes():
            sval = self.state_value_dict[_n]
            self.org_graph.add_state_attribute(_n, "val", [sval if INT_MIN_VAL < sval < INT_MAX_VAL else 'inf'])


class HopefulPermissiveValueIteration(PermissiveValueIteration):

    def __init__(self, game: TwoPlayerGraph, competitive: bool = False, int_val: bool = True, prior_val_vector: Optional[dict] = None):
        super().__init__(game, competitive, int_val, prior_val_vector)
        self._adv_val_vector: Optional[ndarray] = self._val_vector
        self._coop_val_vector: Optional[ndarray] = self._val_vector
    
    @property
    def adv_val_vector(self):
        return self._adv_val_vector
    
    @property
    def coop_val_vector(self):
        return self._coop_val_vector
    

    def extract_strategy(self) -> Tuple[dict, dict]:
        """
         Override the base method to reason over the adv_val_vector.
        """
        _env_str_dict = {}
        _sys_str_dict = {}
        for _n in self.org_graph._graph.nodes():
            # get the max value
            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                _next_max_node = self._get_max_env_val(_n, self.adv_val_vector[:, -1])
                if _next_max_node is not None:
                    _env_str_dict[_n] = _next_max_node
            
            # get the min value
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                _next_min_node = self._get_min_sys_val(node=_n)
                if _next_min_node is not None:
                    _sys_str_dict[_n] = _next_min_node 
        
        return _sys_str_dict, _env_str_dict
    

    def _get_min_sys_val(self, node: Union[str, tuple]) -> Union[List[str], None]:
        """
        Override the base method to compute acVal, i.e., the strategy with the optimal cooperative value.
        :param node: The current node in in the graph
        :param pre_vec: The previous value vector
        :return: A List of states
        """
        adv_succ_vals: List = []
        coop_succ_vals: List = []
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            adv_val = self.adv_val_vector[:, -1][_node_int] + self.org_graph.get_edge_weight(node, _next_n)
            coop_val = self.coop_val_vector[:, -1][_node_int] + self.org_graph.get_edge_weight(node, _next_n)
            if adv_val != math.inf:
                adv_succ_vals.append((_next_n, adv_val))
                coop_succ_vals.append((_next_n, coop_val))
        try:
            _, min_val = min(adv_succ_vals, key=operator.itemgetter(1))
            opt_adv_states = [_node for _node, _node_val in adv_succ_vals if min_val == _node_val]
            adv_coop_vals = [_coop_node_val for _node, _coop_node_val in coop_succ_vals if _node in opt_adv_states]
            opt_adv_coop_val = min(adv_coop_vals)
            return [_node for _node, _node_val in coop_succ_vals if opt_adv_coop_val == _node_val and _node in opt_adv_states]
        except ValueError:
            return None


    def update_state_values(self, adv_val_vector: ndarray, coop_val_vector: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Overide base method to update value vector for both the adv value and the cooperative value. 
        """

        adv_val_pre = copy.copy(adv_val_vector)
        coop_val_pre = copy.copy(coop_val_vector)

        for _n in self.org_graph._graph.nodes():
            _int_node = self.node_int_map[_n]

            if _n in self._accp_states and self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                continue
            
            adv_val_vector[_int_node][0], coop_val_vector[_int_node][0] = self._get_opt_val(node=_n, adv_pre_vec=adv_val_pre, coop_pre_vec=coop_val_pre)

        self._adv_val_vector = np.append(self.adv_val_vector, adv_val_vector, axis=1)
        self._coop_val_vector = np.append(self.coop_val_vector, coop_val_vector, axis=1)

        return adv_val_vector, coop_val_vector


    def solve(self, debug: bool = False, plot: bool = False, extract_strategy: bool = True):
        """
            Overides the base method to run min-max  and min-min Value Iteration inside the loop.
        :param debug:
        :param plot:
        :return:
        """

        # initially in the org val_vector the target node(s) will value 0
        adv_val_vector = copy.deepcopy(self.adv_val_vector)
        adv_val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        # initiate 
        coop_val_vector = copy.deepcopy(self.coop_val_vector)
        coop_val_pre = np.full(shape=(self.num_of_nodes, 1), fill_value=math.inf)

        iter_var = 0

        while not (self._is_same(adv_val_pre, adv_val_vector) and self._is_same(coop_val_pre, coop_val_vector)):
            if debug:
                # if iter_var % 1000 == 0:
                print(f"{iter_var} Iterations")

            adv_val_pre = copy.copy(adv_val_vector)
            coop_val_pre = copy.copy(coop_val_vector)
            iter_var += 1

            # Adversarial perform one step Value Iteration 
            adv_val_vector, coop_val_vector = self.update_state_values(adv_val_vector=adv_val_vector, coop_val_vector=coop_val_vector)

        # safely convert values in the last col of val vector to ints
        if self._int_val:
            adv_int_val_vector = self.adv_val_vector[:, -1].astype(int)
            # coop_int_val_vector = self.coop_val_vector[:, -1].astype(int)
        else:
            adv_int_val_vector = self.adv_val_vector[:, -1]
            # coop_int_val_vector = self.coop_val_vector[:, -1]

        # update the state value dict
        for i in range(self.num_of_nodes):
            _s = self.node_int_map.inverse[i]
            self.state_value_dict.update({_s: adv_int_val_vector[i] if INT_MIN_VAL <  adv_int_val_vector[i] < INT_MAX_VAL  else math.inf})

        # extract sys and env strategy after converging.
        if extract_strategy:
            self._sys_str_dict, self._env_str_dict = self.extract_strategy()

        self._str_dict = {**self._sys_str_dict, **self._env_str_dict}
        self._sys_winning_region = set(self._sys_str_dict.keys()) #.union(self._accp_states)

        if plot:
            self.plot_graph()

        if debug:
            print(f"Number of iteration to converge: {iter_var}")
            # print(f"Init state value: {self.state_value_dict[_init_node]}")
            # self._sanity_check()

    def _get_max_env_val(self, node: Union[str, tuple], pre_vec: ndarray) -> List[str]:
        """
        A method that returns the max value for the current node that belongs to the env.
        :param node: The current node in in the graph
        :param pre_vec: The previous value vector
        :return: A List of states
        """

        _succ_vals: List[tuple] = []
        _only_vals = set({})
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = pre_vec[_node_int] + self.org_graph.get_edge_weight(node, _next_n)
            _only_vals.add(_val)
            _succ_vals.append((_next_n, _val))

        # get org node int value
        if self.competitive:
            _lonly_vals = list(_only_vals)
            if math.inf in _lonly_vals:
                if len(_lonly_vals) == 1 and _lonly_vals[0] == math.inf:
                    _, _val = max(_succ_vals, key=operator.itemgetter(1))
                # if the succ state are all NOT inf
                elif len(_lonly_vals) == 1 and _lonly_vals[0] != math.inf:
                    _val = _lonly_vals[0]
                else:
                    # get the 2nd highest value
                    _lonly_vals.sort()
                    _val = _lonly_vals[-2]
            else:
                _, _val = max(_succ_vals, key=operator.itemgetter(1))
        else:
            _, _val = min(_succ_vals, key=operator.itemgetter(1))
            if _val == math.inf:
                return None

        return [_node for _node, _node_val in _succ_vals if _val == _node_val]

    def _get_opt_val(self, node: Union[str, tuple], adv_pre_vec: ndarray, coop_pre_vec: ndarray) -> Tuple[int, int]:
        """
         This method only return the max (adverasrial)/min (cooperative) value
        """
        adv_succ_vals = set({})
        coop_succ_vals = set({})
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            adv_val = (self.org_graph.get_edge_weight(node, _next_n) + adv_pre_vec[_node_int][0])
            coop_val = (self.org_graph.get_edge_weight(node, _next_n) + coop_pre_vec[_node_int][0])
            adv_succ_vals.add(adv_val)
            coop_succ_vals.add(coop_val)
        
        if self.org_graph.get_state_w_attribute(node, "player") == "eve":
            return min(adv_succ_vals) , min(coop_succ_vals)
        else:
            # Computation for adversarial human
            _lvals = list(adv_succ_vals)
            if math.inf in _lvals:
                if len(_lvals) == 1 and _lvals[0] == math.inf:
                    adv_val = max(adv_succ_vals)
                # if the succ state are all NOT vals
                elif len(_lvals) == 1 and _lvals[0] != math.inf:
                    adv_val = _lvals[0]
                else:
                    # get the 2nd highest value
                    _lvals.sort()
                    adv_val = _lvals[-2]
            else:
                adv_val = max(adv_succ_vals)
            # computation for cooperative human    
            coop_val = min(coop_succ_vals)

            return adv_val, coop_val


class PermissiveCoopValueIteration(ValueIteration):
    """
     Override base method to compute the maximally persmissive set of Cooperative Strategy. 
      Note: It is NOT the set of all OPTIMAL strategies rather it the set of all strategy under which you reach goal state while playing memorylessly. 

      Such strategies are also called non-deferring strategies. 
      
      See: "Synthesis of Maximally Permissive Strategies for LTLf Speciﬁcations", Shufang Zhu and G. D. Giacomo, IJCAI 22, for more details.  
    """
    def __init__(self, game: TwoPlayerGraph, int_val: bool = True, prior_val_vector: Optional[dict] = None):
        super().__init__(game=game, int_val=int_val, competitive=False, prior_val_vector=prior_val_vector)
        self._sys_coop_opt_str_dict: Dict = defaultdict(lambda: -1)

    @property
    def sys_coop_opt_str_dict(self):
        return self._sys_coop_opt_str_dict
    

    def extract_strategy(self) -> Tuple[dict, dict]:
        """
         Overide the base method as get_min_sys_val return Coop and Coop Optimal strategy.
        """
        _env_str_dict = {}
        _sys_str_dict = {}
        for _n in self.org_graph._graph.nodes():
            # get the max value
            if self.org_graph.get_state_w_attribute(_n, "player") == "adam":
                _next_max_node = self._get_max_env_val(_n, self.val_vector[:, -1])
                if _next_max_node is not None:
                    _env_str_dict[_n] = _next_max_node
            
            # get the min value
            elif self.org_graph.get_state_w_attribute(_n, "player") == "eve":
                _next_min_node, opt_next_min_nodes = self._get_min_sys_val(_n, self.val_vector[:, -1])
                if _next_min_node is not None:
                    _sys_str_dict[_n] = _next_min_node
                    self._sys_coop_opt_str_dict[_n] =  opt_next_min_nodes
        
        return _sys_str_dict, _env_str_dict
    

    def _get_min_sys_val(self,  node: Union[str, tuple], pre_vec: ndarray) -> Tuple[Union[str], str]:
        """
         A method that returns the set of nodes along the non-deferring strategies.
            :param node: The current node in the graph
            :param pre_vec: The previous value vector
         :return: The optimal state(s) to transition to if successor states values are not Inf.
        """
        _succ_vals: List[Tuple[str, int]] = []
        _succ_nodes = []
        cval_curr_node =   pre_vec[self.node_int_map[node]]
        for _next_n in self.org_graph._graph.successors(node):
            _node_int = self.node_int_map[_next_n]
            _val = self.org_graph.get_edge_weight(node, _next_n) + pre_vec[_node_int]
            if _val != math.inf:
                _succ_vals.append((_next_n, _val))
            if pre_vec[_node_int] < cval_curr_node:
                _succ_nodes.append(_next_n)
        
        try:
            _, min_val = min(_succ_vals, key=operator.itemgetter(1))
            opt_succ_node = [s for s, v in _succ_vals if v == min_val]
        except ValueError: 
            opt_succ_node = None
        
        if len(_succ_nodes) == 0:
            return None, opt_succ_node
        
        return _succ_nodes, opt_succ_node