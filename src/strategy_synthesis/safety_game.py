
import warnings

from typing import Set
from collections import defaultdict, deque

from .adversarial_game import ReachabilityGame
from ..graph import TwoPlayerGraph, graph_factory


class SafetyGame(ReachabilityGame):
    """
     This class implements a safety game. For a give set of goal state, this class return
      1. Safe set - the set of states from which Sys player can any strategy (if it is non-empty)
      2. Safe Winning str - All action from a Sys player state that belong to the safe set is a safe str.
    
     We play safety game as a complement of reachability game, i.e., 
        If Sys player has a safety objective to stay in F, the Env player has reachability objective to reach S\F.
    """

    def __init__(self, game: TwoPlayerGraph, target_states, debug: bool = False, **kwargs):
        super().__init__(game, debug, **kwargs)
        self._target_states = target_states
    
    @property
    def target_states(self):
        return self._target_states

    @target_states.setter
    def target_states(self, states: Set):
        self._target_states = states
    

    def _sanity_check_player(self):
        """
         A helper function check if very state has a player or not
        """
        for _n in self.game._graph.nodes.data("player"):
            if _n[1] is None:
                warnings.warn(f"Please ensure all nodes have a player attribute. Node {_n} does not have one.")
            
            if not self.game._finite and len(list(self.game._graph.successors(_n[0]))) == 0:
                warnings.warn(f"Please ensure all nodes have at-least one successor. Node {_n} does not have one.")
    
    def _sanity_check_total(self, debug: bool = False):
        """
         Override base method. This method checks if every node has at-least one outgoing edge. Added this check to _sanity_check_player() method.
        """
        pass

    def reachability_solver(self):
        self.solve()

    def solve(self):
        """
         Implements the Safety solver. 

         TODO: Compute
        """
        num_out_edges = self._compute_no_of_node_successors()

        queue = deque()

        _regions = defaultdict(lambda: -1)
        env_winning_region: set = set({})
        sys_str = defaultdict(lambda: set())
        env_str = defaultdict(lambda: -1)

        # Env player's target states: S\F
        env_target_states: set = set(self.game._graph.nodes()).difference(self.target_states)

        for _s in env_target_states:
            # if self.game.get_state_w_attribute(_s, "player") == "adam":
            queue.append(_s)
            _regions[_s] = "adam"
            env_winning_region.add(_s)
        
        while queue:
            _s = queue.popleft()

            for _pre_s in self.game._graph.predecessors(_s):
                # the state has not been explored before
                if _regions[_pre_s] == -1:
                    # if there exists a transition from Env player state to env_target_states
                    if self.game._graph.nodes[_pre_s].get("player") == "adam":
                        queue.append(_pre_s)
                        _regions[_pre_s] = "adam"
                        env_winning_region.add(_pre_s)
                        env_str[_pre_s] = _s

                    # if all transitions from Sys player state transit to env_target_states
                    elif self.game._graph.nodes[_pre_s].get("player") == "eve":
                        num_out_edges[_pre_s] -= 1
                        if num_out_edges[_pre_s] == 0:
                            queue.append(_pre_s)
                            _regions[_pre_s] = "adam"
                            env_winning_region.add(_pre_s)

                    else:
                        warnings.warn(f"Please make sure that every node in the game is assigned a player."
                                        f"Currently node {_pre_s} does not have a player assigned to it")
        
        # for Sys player's winning region
        sys_winning_region: set = set(self.game._graph.nodes()).difference(env_winning_region)

        # Sys can play any strategy in the safe set
        for _s in sys_winning_region:
            if self.game._graph.nodes[_s]["player"] == "eve":
                for _successor in self.game._graph.successors(_s):
                    if _regions[_successor] != "adam":
                        sys_str[_s] = sys_str[_s].union(set([_successor])) 
        

        for _s in env_winning_region:
            if self.game._graph.nodes[_s].get("player") == "adam":
                env_str[_s] = [_succ_s for _succ_s in self.game._graph.successors(_s) if _regions[_succ_s] == "adam"]
        
        self._sys_str = sys_str
        self._env_str = env_str
        self._sys_winning_region = sys_winning_region
        self._env_winning_region = env_winning_region



