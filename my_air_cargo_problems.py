from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr, Expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # DONE create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        from itertools import product
        def generate_concrete_actions(schema: Action, *args: iter):
            '''
            Helper method to generate concrete Actions from Action Schemas.

            :param schema: Action schema to generate concrete actions for
            :param args: List of lists of possible argument values for each variable in action schema, presented in same
            order as variables are introduced in the Action schema.

            :return: List of concrete actions for all possible unique combinations of arguments.
            '''
            args = [a for a in list(product(*args)) if len(a) == len(set(a))]
            concrete_actions = []
            for arg in args:
                ground_args = list(map(expr, arg))
                action = schema.substitute(Expr(schema.name, *schema.args), ground_args)
                precond_pos = [schema.substitute(precond, ground_args) for precond in schema.precond_pos]
                precond_neg = [schema.substitute(precond, ground_args) for precond in schema.precond_neg]
                effect_add = [schema.substitute(effect, ground_args) for effect in schema.effect_add]
                effect_rem = [schema.substitute(effect, ground_args) for effect in schema.effect_rem]
                concrete_actions.append(Action(action, [precond_pos, precond_neg], [effect_add, effect_rem]))
            return concrete_actions

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            # DONE create all load ground actions from the domain Load action
            load_schema = Action(expr("Load(c, p, a)"),
                                 [[expr("At(c, a)"), expr("At(p, a)")], []],    # Preconditions
                                 [[expr("In(c, p)")],                           # Add Effects
                                  [expr("At(c, a)")]])                          # Remove Effects

            return generate_concrete_actions(load_schema, self.cargos, self.planes, self.airports)

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            # DONE create all Unload ground actions from the domain Unload action
            unload_schema = Action(expr("Unload(c, p, a)"),
                                   [[expr("In(c, p)"), expr("At(p, a)")], []],  # Preconditions
                                   [[expr("At(c, a)")],                         # Add Effects
                                    [expr("In(c, p)")]])                        # Remove Effects
            return generate_concrete_actions(unload_schema, self.cargos, self.planes, self.airports)

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()



    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # DONE implement


        # TODO make decode_state more efficient
        model = decode_state(state, self.state_map)
        return [action for action in self.actions_list if action_possible(action, model)]


    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # DONE implement
        assert action_possible(action, decode_state(state, self.state_map)), "Action must be one of self.actions(state)"

        new_state = list(state)
        for idx, literal in enumerate(self.state_map):
            if literal in action.effect_add:
                new_state[idx] = 'T'
            elif literal in action.effect_rem:
                new_state[idx] = 'F'

        return ''.join(new_state)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # DONE implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        #  Because the goals in all our problems are conjunctions of clauses of the form:
        #  At(cargo, airport)
        #  And because the only action - Unload(cargo, airport) - able to add such a goal literal to the current state:
        #  1) Cannot remove any goal literal from the state (as the unload action never removes an At literal)
        #  2) Only adds at most a single At literal to the current state
        #  We are justified in making the independent subgoal assumption and so the ignore_preconditions heuristic is
        #  simply the current number of unsatisfied goal conditions.  (as per Russell-Norvig Ed-3 10.2.3 TODO check this)
        #
        #  (i.e. the minimum number of actions required is specifically the minimum number of unload actions required).

        model = decode_state(node.state, self.state_map).pos
        count = len([literal for literal in self.goal if literal not in model])
        return count

def action_possible(action, model):
    # An action is possible in a state if all the action's preconditions are met by the state
    return all(precond in model.pos for precond in action.precond_pos) and \
           all(precond in model.neg for precond in action.precond_neg)  # 2nd clause not strictly necessary
                                                                        # as ACP actions don't have neg preconds


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    pass


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    pass
