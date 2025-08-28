import logging, random
from typing import Any, List, Optional, Tuple, Callable

from .functional_computers import *
from .types_computers import QuestionSpec, GraphSpec, NodeSpec
from .generate_graph_computers import SystemNodeProperties, OtherProperties

logger = logging.getLogger(__name__)

# --- Placeholders ---
class SystemNode(FunctionalOperator): pass
class FakeNodeName(FunctionalOperator): pass
class Status(FunctionalOperator): pass
class SecurityLevel(FunctionalOperator): pass
class LocationSector(FunctionalOperator): pass
class FirmwareVersion(FunctionalOperator): pass

# --- Argument selectors ---
class ArgumentSelector:
    def get(self, graph: GraphSpec) -> Any: raise NotImplementedError

class SelectSystemNode(ArgumentSelector):
    def get(self, graph):
        return random.choice(list(graph.nodes.values()))

class SelectFakeNodeName(ArgumentSelector):
    def get(self, graph):
        names = {n.name for n in graph.nodes.values()}
        for _ in range(100):
            nm = f"{random.choice(OtherProperties['name_prefix'])}_{random.choice(OtherProperties['name_suffix'])}-{random.randint(1000,9999)}"
            nm = nm.title()
            if nm not in names: return nm
        return f"FakeNode_{random.randint(1000,9999)}"

class SelectStatus(ArgumentSelector):
    def get(self, graph): return random.choice(SystemNodeProperties["status"])
class SelectSecurityLevel(ArgumentSelector):
    def get(self, graph): return random.choice(SystemNodeProperties["security_level"])
class SelectLocationSector(ArgumentSelector):
    def get(self, graph): return random.choice(SystemNodeProperties["location_sector"])
class SelectFirmwareVersion(ArgumentSelector):
    def get(self, graph): return random.choice(SystemNodeProperties["firmware_version"])

class SelectAdjacentSystemPair(ArgumentSelector):
    def get(self, graph: GraphSpec) -> Tuple[NodeSpec, NodeSpec]:
        if not graph.edges:
            raise ValueError("Cannot select edge from empty graph")
        edge = random.choice(graph.edges)
        node1 = graph.nodes[edge.station1]
        node2 = graph.nodes[edge.station2]
        self.value = (node1, node2)
        # print(node1, node2)
        return self.value    
    @property
    def args(self):
        return list(self.value) if hasattr(self, "value") else []


ARG_SELECTORS = {
    SystemNode: SelectSystemNode(),
    FakeNodeName: SelectFakeNodeName(),
    Status: SelectStatus(),
    SecurityLevel: SelectSecurityLevel(),
    LocationSector: SelectLocationSector(),
    FirmwareVersion: SelectFirmwareVersion(),
    (SystemNode, SystemNode, "adjacent"): SelectAdjacentSystemPair(),
}
# --- QuestionForm (unchanged) ---
class QuestionForm:
    def __init__(
        self,
        placeholders: List[type],  # Use classes like Station, Line, etc.
        english: str,
        functional_builder: Callable[
            ..., FunctionalOperator
        ],  # A function taking selected args and returning a FunctionalOperator
        type_string: str,
        arguments_valid: Callable[..., bool] = (lambda *args: True),
        answer_valid: Callable[..., bool] = (lambda *args: True),
        group: Optional[str] = None,
        subgroup: Optional[str] = None,
        type_id: Optional[int] = None,
    ):
        self.placeholders = placeholders
        self.english = english
        self.functional_builder = functional_builder
        self.type_id = type_id
        self.type_string = type_string
        self.arguments_valid = arguments_valid
        self.answer_valid = answer_valid
        self.group = group
        self.subgroup = subgroup if subgroup else "Fact"

    def __repr__(self):
        return f"QuestionForm({self.type_string}: '{self.english_explain()}')"

    def english_explain(self):
        # Format with placeholder names like {Station}, {Line}
        ph_names = [f"{{{i.__name__}}}" for i in self.placeholders]
        try:
            return self.english.format(*ph_names)
        except IndexError:
            # Handle cases where format string doesn't match placeholders
            return f"{self.english} (placeholders: {ph_names})"

    def generate(
        self, graph: GraphSpec, runtime_args: Any
    ) -> Tuple[Optional[QuestionSpec], Optional[Any]]:
        """Generates a QuestionSpec and its answer for the given graph."""
        try:
            # 1. Select concrete arguments using selectors
            selected_args = []
            if self.placeholders and isinstance(self.placeholders[-1], str):
                selector = ARG_SELECTORS.get(tuple(self.placeholders))
                if not selector:
                    raise TypeError(
                        f"No argument selector defined for placeholder type: tuple"
                    )
                selected_args = list(selector.get(graph))
            else:
                for ph_type in self.placeholders:
                    selector = ARG_SELECTORS.get(ph_type)
                    if not selector:
                        raise TypeError(
                            f"No argument selector defined for placeholder type: {ph_type.__name__}"
                        )
                    selected_args.append(selector.get(graph))
            # 2. Validate selected arguments
            if not self.arguments_valid(graph, *selected_args):
                logger.debug(
                    f"Arguments invalid for {self.type_string}: {selected_args}"
                )
                return None, None  # Invalid arguments, skip this attempt

            # 3. Build the functional program using the *selected arguments*
            functional_program = self.functional_builder(*selected_args)

            # 4. Execute the functional program to get the answer
            answer = functional_program(graph)

            self.answer_type = type(answer)

            # 5. Validate the answer
            if not self.answer_valid(graph, answer, *selected_args):
                logger.debug(
                    f"Answer invalid for {self.type_string}: {answer} (args: {selected_args})"
                )
                return None, None  # Invalid answer, skip this attempt

            # 6. Format the English question
            english_args = []
            for arg in selected_args:
                if isinstance(arg, (NodeSpec,)):
                    english_args.append(arg.name)
                elif isinstance(arg, str):
                    english_args.append(arg)
                else:
                    english_args.append(str(arg))
            english = self.english.format(*english_args)

            # 7. Get the serializable functional representation
            functional_dict = functional_program.to_dict()

            # 8. Create the QuestionSpec (Cypher is removed)
            q_spec = QuestionSpec(
                english=english,
                functional=functional_dict,
                type_id=self.type_id,
                type_string=self.type_string,
                group=self.group,
                subgroup=self.subgroup,
            )

            return q_spec, answer

        except ValueError as ve:
            logger.debug(
                f"Generation failed for '{self.english_explain()}' due to ValueError: {ve}"
            )
            return None, None  # Signal failure for this attempt
        except (TypeError, AttributeError, KeyError, IndexError) as e:
            logger.error(
                f"Error generating question '{self.english_explain()}': {e}",
                exc_info=True,
            )
            return None, None  # Signal failure
        except Exception as e:
            logger.error(
                f"Unexpected error generating question '{self.english_explain()}': {e}",
                exc_info=True,
            )
            return None, None  # Signal failure

# --- Questions for System Grid ---
question_forms: List[QuestionForm] = []

ONE_WORD = "\n\nAnswer directly: "
LIST = "\n\nOutput a comma-separated list: "
BOOL = "\n\nAnswer with 'True' or 'False':\n\nAnswer: "
COUNT = "\n\nAnswer with a number:\n\nAnswer: "
CYCLE = "\n\nAnswer with 'True' if it is in a cycle, otherwise 'False':\n\nAnswer: "

question_forms.extend([
    # Fact-based
    QuestionForm([SystemNode],
        "What is the status of node {}?" + ONE_WORD,
        lambda s: Pick(s, "status"), "NodePropertyStatus", group="FactBased"),
    QuestionForm([SystemNode],
        "What is the security level of node {}?" + ONE_WORD,
        lambda s: Pick(s, "security_level"), "NodePropertySecurity", group="FactBased"),
    QuestionForm([SystemNode],
        "Which sector is node {} located in?" + ONE_WORD,
        lambda s: Pick(s, "location_sector"), "NodePropertyLocation", group="FactBased"),
    QuestionForm([SystemNode],
        "What firmware version runs on {}?" + ONE_WORD,
        lambda s: Pick(s, "firmware_version"), "NodePropertyFirmware", group="FactBased"),
    QuestionForm([SystemNode],
        "How many power units does {} consume?" + COUNT,
        lambda s: Pick(s, "power_consumption_units"), "NodePropertyPower", group="FactBased"),

    QuestionForm([SystemNode],
        "Is there a node named {} in the grid?" + BOOL,
        lambda a: Const(True), "NodeExistence1", group="FactBased"),
    QuestionForm([FakeNodeName],
        "Is there a node named {} in the grid?" + BOOL,
        lambda a: Const(False), "NodeExistence2", group="FactBased"),

    QuestionForm([SystemNode, SystemNode, "adjacent"],
        "Are nodes {} and {} directly linked?" + BOOL,
        lambda a, b: Adjacent(a, b), "NodeAdjacentTrue",
        group="FactBased", arguments_valid=lambda g,a,b: a.id!=b.id),
    
    QuestionForm([SystemNode, SystemNode],
        "Are nodes {} and {} directly linked?" + BOOL,
        lambda a, b: Adjacent(a, b), "NodeAdjacent",
        group="FactBased", arguments_valid=lambda g,a,b: a.id!=b.id),

    # Reasoning – Aggregation
    QuestionForm([Status],
        "How many nodes have status '{}'?" + COUNT,
        lambda v: Count(Filter(AllNodes(), "status", v)),
        "CountNodesWithStatus", group="ReasoningBased", subgroup="Aggregation"),
    QuestionForm([LocationSector],
        "List all nodes in {}." + LIST,
        lambda v: Pluck(Filter(AllNodes(), "location_sector", v), "name"),
        "ListNodesInSector", group="ReasoningBased", subgroup="Aggregation"),
    QuestionForm([],
        "What is the most common firmware version?" + ONE_WORD,
        lambda: Mode(Pluck(AllNodes(), "firmware_version")),
        "MostCommonFirmware", group="ReasoningBased", subgroup="Aggregation"),

    # Reasoning – Filtering
    QuestionForm([LocationSector, SecurityLevel],
        "How many nodes in {} have security level '{}'?" + COUNT,
        lambda s,l: Count(Filter(Filter(AllNodes(), "location_sector", s), "security_level", l)),
        "CountNodesWithTwoProps", group="ReasoningBased", subgroup="Filter"),
    QuestionForm([SystemNode],
        "How many neighbors of {} are 'Operational'?" + COUNT,
        lambda n: Count(Filter(Neighbors(n), "status", "Operational")),
        "CountNeighborsOperational", group="ReasoningBased", subgroup="Filter"),

    # Reasoning – Path & Topology
    QuestionForm([SystemNode, SystemNode],
        "How many nodes are on shortest path between {} and {}?" + COUNT,
        lambda a,b: Count(ShortestPath(a, b, [])),
        "ShortestPathLen", group="ReasoningBased", subgroup="PathReasoning",
        arguments_valid=lambda g,a,b: a.id!=b.id),
    QuestionForm([SystemNode, SystemNode],
        "How many nodes lie between {} and {} on that path?" + COUNT,
        lambda a,b: Max(Subtract(Count(ShortestPath(a,b,[])),2),0),
        "NodesBetween", group="ReasoningBased", subgroup="PathReasoning",
        arguments_valid=lambda g,a,b: a.id!=b.id),
    QuestionForm([SystemNode, SystemNode, Status],
        "Is there a path from {} to {} avoiding status '{}'?" + BOOL,
        lambda a,b,s: NotEmpty(
            ShortestPathOnlyUsing(a,b,Without(AllNodes(),"status",s),[])
        ),
        "PathAvoidingStatus", group="ReasoningBased", subgroup="PathReasoning",
        arguments_valid=lambda g,a,b,s: a.id!=b.id),

    QuestionForm([SystemNode],
        "How many other nodes are within 3 hops of {}?" + COUNT,
        lambda a: Count(WithinHops(a,3)),
        "WithinHops", group="ReasoningBased", subgroup="Topology"),
    QuestionForm([SystemNode],
        "Is {} part of a cycle?" + CYCLE,
        lambda n: HasCycle(n),
        "HasCycle", group="ReasoningBased", subgroup="Topology"),
    QuestionForm([SystemNode, SystemNode],
        "Are {} and {} connected via exactly one intermediary?" + BOOL,
        lambda a,b: Equal(Count(ShortestPath(a,b,[])),3),
        "OneIntermediary", group="ReasoningBased", subgroup="Topology",
        arguments_valid=lambda g,a,b: a.id!=b.id),
])

# assign type_ids
for i, f in enumerate(question_forms):
    f.type_id = i
logger.info(f"Initialized {len(question_forms)} computer‐grid question forms.")
