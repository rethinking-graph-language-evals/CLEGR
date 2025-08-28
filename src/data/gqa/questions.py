import logging

logger = logging.getLogger(__name__)
import random  # For selecting arguments
from typing import Optional, List, Tuple, Callable, Any
from collections import defaultdict

from .functional import *
from .types_ import QuestionSpec, GraphSpec, NodeSpec, LineSpec
from .generate_graph import StationProperties

# --- Define Group Constants ---
GROUP_FACT_BASED = "FactBased"
GROUP_REASONING_BASED = "ReasoningBased"

# --------------------------------------------------------------------------
# Directory of question types
# --------------------------------------------------------------------------


# Define argument selector classes (simplified from original implicit behavior)
class ArgumentSelector:
    def get(self, graph: GraphSpec) -> Any:
        raise NotImplementedError

    @property
    def args(self):  # Mimic original structure if needed by QuestionForm
        return [self.value] if hasattr(self, "value") else []


class SelectStation(ArgumentSelector):
    def get(self, graph: GraphSpec) -> NodeSpec:
        if not graph.nodes:
            raise ValueError("Cannot select station from empty graph")
        self.value = random.choice(list(graph.nodes.values()))
        return self.value


class SelectLine(ArgumentSelector):
    def get(self, graph: GraphSpec) -> LineSpec:
        if not graph.lines:
            raise ValueError("Cannot select line from empty graph")
        self.value = random.choice(list(graph.lines.values()))
        return self.value


class SelectFakeStationName(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        # Generate a name unlikely to exist (based on original logic)
        actual_station_names = {str(node.name) for node in graph.nodes.values()}
        max_stn_range = (
            len(graph.nodes) * 3
        )  # Increase range for better chance of fake name
        attempts = 0
        while attempts < 10:
            fake_name = str(random.randint(len(graph.nodes) + 1, max_stn_range))
            if fake_name not in actual_station_names:
                self.value = fake_name
                return self.value
            attempts += 1
        # Fallback if integer names clash too much
        self.value = f"FakeStation_{random.randint(1000, 9999)}"
        return self.value


class SelectArchitecture(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        self.value = random.choice(StationProperties["architecture"])
        return self.value


class SelectSize(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        self.value = random.choice(StationProperties["size"])
        return self.value


class SelectMusic(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        self.value = random.choice(StationProperties["music"])
        return self.value


class SelectCleanliness(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        self.value = random.choice(StationProperties["cleanliness"])
        return self.value


class SelectColor(ArgumentSelector):
    def get(self, graph: GraphSpec) -> str:
        # Collect all unique colors from the lines in the graph
        colors = [line.color for line in graph.lines.values()]
        self.value = random.choice(colors)
        return self.value

class SelectAdjacentStationPair(ArgumentSelector):
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

class SelectLineStationPair(ArgumentSelector):
    def get(self, graph: GraphSpec) -> Tuple[NodeSpec, NodeSpec]:
        if not graph.lines:
            raise ValueError("Cannot select from empty graph")
        
        attempts = 0
        while attempts < 10:
            # Randomly select a line
            line = random.choice(list(graph.lines.values()))
            line_edges = [edge for edge in graph.edges if edge.line_id == line.id]
            station_ids = set()
            for edge in line_edges:
                station_ids.add(edge.station1)
                station_ids.add(edge.station2)
            station_ids = list(station_ids)
            if len(station_ids) >= 2:
                # Pick two distinct station IDs
                station1_id, station2_id = random.sample(station_ids, 2)
                station1 = graph.nodes[station1_id]
                station2 = graph.nodes[station2_id]
                self.value = (station1, station2)
                return self.value
            attempts += 1
        
        raise ValueError("Failed to find a line with at least two stations")
    
    @property
    def args(self):
        return list(self.value) if hasattr(self, "value") else []

class SelectCommonStationPair(ArgumentSelector):
    def get(self, graph: GraphSpec) -> Tuple[NodeSpec, NodeSpec]:
        if not graph.edges or not graph.nodes:
            raise ValueError("Graph must have nodes and edges")

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in graph.edges:
            adjacency[edge.station1].add(edge.station2)
            adjacency[edge.station2].add(edge.station1)

        attempts = 0
        while attempts < 10:
            # Randomly pick a central station
            central_station_id = random.choice(list(graph.nodes.keys()))
            neighbors = list(adjacency[central_station_id])

            if len(neighbors) >= 2:
                # Choose two neighbors
                neighbor1_id, neighbor2_id = random.sample(neighbors, 2)
                self.value = (
                    graph.nodes[neighbor1_id],
                    graph.nodes[neighbor2_id],
                )
                return self.value

            attempts += 1

        raise ValueError("Failed to find a station with at least two adjacent stations")
    
    @property
    def args(self):
        return list(self.value) if hasattr(self, "value") else []




# Map original classes like Station, Line to these selectors
ARG_SELECTORS = {
    Station: SelectStation(),
    Line: SelectLine(),
    FakeStationName: SelectFakeStationName(),
    Architecture: SelectArchitecture(),
    Size: SelectSize(),
    Music: SelectMusic(),
    Cleanliness: SelectCleanliness(),
    Color: SelectColor(),
    (Station, Station, "adjacent"): SelectAdjacentStationPair(),
    (Station, Station, "line"): SelectLineStationPair(),
    (Station, Station, "common"): SelectCommonStationPair(),
    # Boolean selector if needed
}


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
        robustness: Optional[Callable] = None,
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
        self.robustness = robustness

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
        self, graph: GraphSpec, runtime_args: Any, robust: Optional[bool] = False
    ) -> Tuple[Optional[QuestionSpec], Optional[Any], Optional[GraphSpec]]:
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
                # print(f"Arguments invalid for {self.type_string}: {selected_args}")
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
                if isinstance(arg, (NodeSpec, LineSpec)):
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
            graph_robust = None
            if robust and self.robustness:
                graph_robust = self.robustness(graph, selected_args)                

                return q_spec, answer, graph_robust
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


# ==========================================================================
# Question Definitions
# ==========================================================================

question_forms: List[QuestionForm] = []

# Line-level aggregation/filtering
get_line_nodes = lambda l: Unique(Nodes(Filter(AllEdges(), "line_id", Pick(l, "id"))))

# --- Suffixes ---
ONE_WORD_SUFFIX = "\n\nAnswer directly: "
LIST_SUFFIX = "\n\nOutput a comma-separated list: "
BOOL_SUFFIX = "\n\nAnswer with 'True' or 'False':\n\nAnswer: "
COUNT_SUFFIX = "\n\nAnswer with a number:\n\nAnswer: "
CYCLE_SUFFIX = "\n\nAnswer with 'True' if it is in a cycle, otherwise 'False':\n\nAnswer: "

# --- Updated QuestionForm Entries ---
question_forms.extend(
    [
        # Fact-based: Station properties
        QuestionForm(
            [Station],
            "How clean is {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "cleanliness"),
            "StationPropertyCleanliness",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "What is the cleanliness level of {} station?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "cleanliness"),
            "StationPropertyCleanliness2",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "How big is {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "size"),
            "StationPropertySize",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "What size is {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "size"),
            "StationPropertySize2",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "What music plays at {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "music"),
            "StationPropertyMusic",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Which type of music is played at {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "music"),
            "StationPropertyMusic2",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "What architectural style is {}?" + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "architecture"),
            "StationPropertyArchitecture",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Describe {} station's architectural style." + ONE_WORD_SUFFIX,
            lambda s: Pick(s, "architecture"),
            "StationPropertyArchitecture2",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Does {} have disabled access?" + BOOL_SUFFIX,
            lambda s: Pick(s, "disabled_access"),
            "StationPropertyDisabledAccess",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Is there disabled access at {}?" + BOOL_SUFFIX,
            lambda s: Pick(s, "disabled_access"),
            "StationPropertyDisabledAccess2",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Does {} have rail connections?" + BOOL_SUFFIX,
            lambda s: Pick(s, "has_rail"),
            "StationPropertyHasRail",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "Can you get rail connections at {}?" + BOOL_SUFFIX,
            lambda s: Pick(s, "has_rail"),
            "StationPropertyHasRail2",
            group=GROUP_FACT_BASED,
        ),
        # Fact-based: Station existence
        QuestionForm(
            [Station],
            "Is there a station called {}?" + BOOL_SUFFIX,
            lambda a: Const(True),
            "StationExistence1",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [FakeStationName],
            "Is there a station called {}?" + BOOL_SUFFIX,
            lambda a: Const(False),
            "StationExistence2",
            group=GROUP_FACT_BASED,
        ),
        # Fact-based: Line info
        QuestionForm(
            [Station],
            "Which lines is {} on?" + LIST_SUFFIX,
            lambda a: Unique(Pluck(Edges(a), "line_name")),
            "StationLine",
            group=GROUP_FACT_BASED,
        ),
        QuestionForm(
            [Station],
            "How many lines is {} on?" + COUNT_SUFFIX,
            lambda a: Count(Unique(Pluck(Edges(a), "line_name"))),
            "StationLineCount",
            group=GROUP_FACT_BASED,
        ),
        # Fact-based: Adjacency
        QuestionForm(
            [Station, Station, "adjacent"],
            "Are {} and {} adjacent?" + BOOL_SUFFIX,
            lambda a, b: Adjacent(a, b),
            "StationAdjacentAlwaysTrue",
            group=GROUP_FACT_BASED,
        ),

        QuestionForm(
            [Station, Station],
            "Are {} and {} adjacent?" + BOOL_SUFFIX,
            lambda a, b: Adjacent(a, b),
            "StationAdjacent",
            group=GROUP_FACT_BASED,
        ),

        # Fact-based: Edge (line) properties between stations (Level 1: Edge Properties)
        QuestionForm(
            [Station, Station, "adjacent"],
            "What color is the line between {} and {}?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Pick(EdgesBetween(s1, s2), "line_color"),
            "EdgePropertyColor",
            group=GROUP_FACT_BASED,
            arguments_valid=lambda g, s1, s2: Adjacent(s1, s2).op(g, s1, s2) and s1.id != s2.id,
        ),
        QuestionForm(
            [Station, Station, "adjacent"],
            "Does the line between {} and {} have air conditioning?" + BOOL_SUFFIX,
            lambda s1, s2: Pick(EdgesBetween(s1, s2), "line_has_aircon"),
            "EdgePropertyAircon",
            group=GROUP_FACT_BASED,
            arguments_valid=lambda g, s1, s2: Adjacent(s1, s2).op(g, s1, s2) and s1.id != s2.id,
        ),
        QuestionForm(
            [Station, Station, "adjacent"],
            "What stroke style is the line between {} and {}?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Pick(EdgesBetween(s1, s2), "line_stroke"),
            "EdgePropertyStroke",
            group=GROUP_FACT_BASED,
            arguments_valid=lambda g, s1, s2: Adjacent(s1, s2).op(g, s1, s2) and s1.id != s2.id,
        ),
        QuestionForm(
            [Station, Station, "adjacent"],
            "When was the line between {} and {} built?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Pick(EdgesBetween(s1, s2), "line_built"),
            "EdgePropertyBuilt",
            group=GROUP_FACT_BASED,
            arguments_valid=lambda g, s1, s2: Adjacent(s1, s2).op(g, s1, s2) and s1.id != s2.id,
        ),
        # Reasoning: Station-level aggregations [34 total]
        QuestionForm(
            [Station, Station],
            "Which station is adjacent to both {} and {}?" + ONE_WORD_SUFFIX,
            lambda a, b: Pick(
                UnpackUnitList(Sample(Intersection(Neighbors(a), Neighbors(b)), 1)),
                "name",
            ),
            "StationPairAdjacent",
            subgroup="Aggregation",
            group=GROUP_REASONING_BASED, #Earlier - GROUP_FACT_BASED
        ),
        QuestionForm(
            [Architecture, Station],
            "Which {} station is adjacent to {}?" + ONE_WORD_SUFFIX,
            lambda arch, station: Pick(
                UnpackUnitList(
                    Sample(Filter(Neighbors(station), "architecture", arch), 1)
                ),
                "name",
            ),
            "StationArchitectureAdjacent",
            subgroup="Aggregation",
            group=GROUP_REASONING_BASED, #Earlier - GROUP_FACT_BASED
        ),
        # Reasoning: Line-level aggregations
        QuestionForm(
            [Line],
            "How many architectural styles does {} pass through?" + COUNT_SUFFIX,
            lambda l: Count(Unique(Pluck(get_line_nodes(l), "architecture"))),
            "LineTotalArchitectureCount",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
        ),
        QuestionForm(
            [Line],
            "How many music styles does {} pass through?" + COUNT_SUFFIX,
            lambda l: Count(Unique(Pluck(get_line_nodes(l), "music"))),
            "LineTotalMusicCount",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
        ),
        QuestionForm(
            [Line],
            "How many sizes of station does {} pass through?" + COUNT_SUFFIX,
            lambda l: Count(Unique(Pluck(get_line_nodes(l), "size"))),
            "LineTotalSizeCount",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
        ),
        # Reasoning: Filtered line questions
        QuestionForm(
            [Music, Line],
            "How many stations playing {} does {} pass through?" + COUNT_SUFFIX,
            lambda v, l: CountIfEqual(Pluck(get_line_nodes(l), "music"), v),
            "LineFilterMusicCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Cleanliness, Line],
            "How many {} stations does {} pass through?" + COUNT_SUFFIX,
            lambda v, l: CountIfEqual(Pluck(get_line_nodes(l), "cleanliness"), v),
            "LineFilterCleanlinessCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Size, Line],
            "How many {} stations does {} pass through?" + COUNT_SUFFIX,
            lambda v, l: CountIfEqual(Pluck(get_line_nodes(l), "size"), v),
            "LineFilterSizeCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Line],
            "How many stations with disabled access does {} pass through?"
            + COUNT_SUFFIX,
            lambda l: CountIfEqual(Pluck(get_line_nodes(l), "disabled_access"), True),
            "LineFilterDisabledAccessCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Line],
            "How many stations with rail connections does {} pass through?"
            + COUNT_SUFFIX,
            lambda l: CountIfEqual(Pluck(get_line_nodes(l), "has_rail"), True),
            "LineFilterHasRailCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Line],
            "Which stations does {} pass through?" + LIST_SUFFIX,
            lambda l: Pluck(get_line_nodes(l), "name"),
            "LineStations",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
        ),
        # Reasoning: Graph traversal
        QuestionForm(
            [Station, Station],
            "How many stations are between {} and {}?" + COUNT_SUFFIX,
            lambda n1, n2:  Max(Subtract(Count(ShortestPath(n1, n2, [])), 2),0),
            "StationShortestCount",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
        ),
        QuestionForm(
            [Station, Station, Cleanliness],
            "How many stations are on the shortest path between {} and {} avoiding {} stations?"
            + COUNT_SUFFIX,
            lambda n1, n2, c: Max(
                Subtract(
                    Count(
                        ShortestPathOnlyUsing(
                            n1, n2, Without(AllNodes(), "cleanliness", c), None
                        )
                    ),
                    2,
                ),
                0,
            ),
            "StationShortestAvoidingCount",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
        ),
        QuestionForm(
            [Station, Station, Architecture],
            "How many stations are on the shortest path between {} and {} avoiding {} architecture stations?"
            + COUNT_SUFFIX,
            lambda n1, n2, c: Max(
                Subtract(
                    Count(
                        ShortestPathOnlyUsing(
                            n1, n2, Without(AllNodes(), "architecture", c), None
                        )
                    ),
                    2,
                ),
                0,
            ),
            "StationShortestAvoidingArchitectureCount",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
        ),
        QuestionForm(
            [Station],
            "How many other stations are two stops or closer to {}?" + COUNT_SUFFIX,
            lambda a: Count(WithinHops(a, 2)),
            "StationTwoHops",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station, Station],
            "How many distinct routes are there between {} and {}?" + COUNT_SUFFIX,
            lambda n1, n2: Count(Paths(n1, n2)),
            "DistinctRoutes",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station],
            "Is {} part of a cycle?" + CYCLE_SUFFIX,
            lambda n1: HasCycle(n1),
            "HasCycle",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station, Station, "common"],
            "Are {} and {} connected by the same station?" + BOOL_SUFFIX,
            lambda a, b: Equal(Count(ShortestPath(a, b, [])), 3),
            "StationOneApartTrue",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station, Station],
            "Are {} and {} connected by the same station?" + BOOL_SUFFIX,
            lambda a, b: Equal(Count(ShortestPath(a, b, [])), 3),
            "StationOneApart",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station, Station, "line"],
            "Are {} and {} on the same line?" + BOOL_SUFFIX,
            lambda a, b: HasIntersection(
                Unique(Pluck(Edges(a), "line_name")),
                Unique(Pluck(Edges(b), "line_name")),
            ),
            "StationSameLineTrue",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        QuestionForm(
            [Station, Station],
            "Are {} and {} on the same line?" + BOOL_SUFFIX,
            lambda a, b: HasIntersection(
                Unique(Pluck(Edges(a), "line_name")),
                Unique(Pluck(Edges(b), "line_name")),
            ),
            "StationSameLineTrue",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
        ),
        # New questions
        QuestionForm(
            [Line, Station],
            "How many stations in {} are of the same size as {}" + COUNT_SUFFIX,
            lambda l, s: CountIfEqual(Pluck(get_line_nodes(l), "size"), Pick(s, "size")),
            "CountEqualSizeStation",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Line, Line],
            "How many stations are shared between the {} and the {}?" + COUNT_SUFFIX,
            lambda l1, l2: Count(Intersection(get_line_nodes(l1), get_line_nodes(l2))),
            "LineIntersectionStations",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
            arguments_valid=lambda g, l1, l2: l1.id != l2.id,
        ),
        QuestionForm(
            [Station, Station, Station],
            "Is {} on the shortest path between {} and {}?" + BOOL_SUFFIX,
            lambda sC, sA, sB: NotEmpty(Intersection(ShortestPath(sA, sB, []), [sC])),
            "NodeOnPath",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
            arguments_valid=lambda g, sC, sA, sB: sA.id != sB.id and sA.id != sC.id and sB.id != sC.id,
        ),
        QuestionForm(
            [Station],
            "What is the most common architectural style of stations within 2 hops of {}?" + ONE_WORD_SUFFIX,
            lambda s: Mode(Pluck(WithinHops(s, 2), "architecture")),
            "TopologyMostCommonArch",
            group=GROUP_REASONING_BASED,
            subgroup="Topology",
            answer_valid=lambda g, ans, s: ans != 'none',
        ),
        QuestionForm(
            [Station, Station],
            "What is the most common music style on the shortest path between {} and {}?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Mode(Pluck(ShortestPath(s1, s2, []), "music")),
            "PathMostCommonMusic",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
            arguments_valid=lambda g, s1, s2: s1.id != s2.id,
            answer_valid=lambda g, ans, s1, s2: ans != 'none',
        ),
        QuestionForm(
            [],
            "How many stations are both large and have disabled access?" + COUNT_SUFFIX,
            lambda: Count(Filter(Filter(AllNodes(), "size", "large"), "disabled_access", True)),
            "CountIntersectionProperties",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        # New Questions 2

        # Reasoning: Edge (line) filtering questions (Level 2: Edge Filtering)
        QuestionForm(
            [Station],
            "How many air-conditioned lines is {} connected to?" + COUNT_SUFFIX,
            lambda s: CountIfEqual(PluckLineProperty(Edges(s), "line_has_aircon"), True),
            "EdgeFilterAirconCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
            [Color, Station],
            "How many {} lines is {} connected to?" + COUNT_SUFFIX,
            lambda c, s: CountIfEqual(PluckLineProperty(Edges(s), "color"), c),
            "EdgeFilterColorCount",
            group=GROUP_REASONING_BASED,
            subgroup="Filter",
        ),
        QuestionForm(
                [Station, Station],
                "How many years newer is the newest line between {} and {} compared to the oldest?" + COUNT_SUFFIX,
                lambda s1, s2: Subtract(
                    Max(Map(PluckLineProperty(PathEdges(ShortestPath(s1, s2, [])), "line_built"), ParseInt)),
                    Min(Map(PluckLineProperty(PathEdges(ShortestPath(s1, s2, [])), "line_built"), ParseInt))
                ),
                "PathYearSpan",
                group=GROUP_REASONING_BASED,
                subgroup="PathReasoning",
                arguments_valid=lambda g, s1, s2: s1.id != s2.id and len(ShortestPath(s1, s2, []).op(g, s1, s2, [])) > 1,

            ),
        # Reasoning: Path Optimization (Level 3: Path Optimization)
        QuestionForm(
            [Station, Station],
            "What is the most common line color on the shortest path between {} and {}?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Mode(PluckLineProperty(PathEdges(ShortestPath(s1, s2, [])), "color")),
            "PathOptimalColor",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
            arguments_valid=lambda g, s1, s2: s1.id != s2.id,
            answer_valid=lambda g, ans, s1, s2: ans != 'none',
        ),
        QuestionForm(
            [Station, Station],
            "What is the earliest year a line was built on the shortest path between {} and {}?" + ONE_WORD_SUFFIX,
            lambda s1, s2: Min(Map(PluckLineProperty(PathEdges(ShortestPath(s1, s2, [])), "line_built"), ParseInt)),
            "PathEarliestBuilt",
            group=GROUP_REASONING_BASED,
            subgroup="PathReasoning",
            arguments_valid=lambda g, s1, s2: s1.id != s2.id,
        ),
        # Reasoning: Comparative Analysis of Paths (Level 3: Comparative Analysis of Paths)
    QuestionForm(
            [Line, Line],
            "Which line has more stations with disabled access, {} or {}?" + ONE_WORD_SUFFIX,
            lambda l1, l2: Pick(
                ArgMax(
                    [l1, l2],
                    lambda graph, l: CountIfEqual(
                        Pluck(GetLineNodes(l), "disabled_access"),
                        True
                    )
                ),
                "name"
            ),
            "CompareLineDisabledAccess",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
            # arguments_valid=lambda g, l1, l2: l1.id != l2.id,
        ),

        QuestionForm(
            [Architecture, Architecture],
            "Which architectural style has more stations, {} or {}?" + ONE_WORD_SUFFIX,
            lambda a1, a2: ArgMax(
                [a1, a2],
                # FIX: The lambda for ArgMax must accept 'graph' and the item ('a').
                lambda graph, a: CountIfEqual(Pluck(AllNodes(), "architecture"), a)
            ),
            "CompareArchitectureCount",
            group=GROUP_REASONING_BASED,
            subgroup="Aggregation",
            arguments_valid=lambda g, a1, a2: a1 != a2,
        ),
    ]
)

# --- Assign type_ids ---
for idx, form in enumerate(question_forms):
    form.type_id = idx

# logger.info(f"Initialized {len(question_forms)} question forms with group assignments.")