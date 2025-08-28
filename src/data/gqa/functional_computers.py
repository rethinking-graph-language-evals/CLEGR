import random
import networkx as nx
from collections import Counter
from inspect import signature

from .types_computers import NodeSpec, EdgeSpec, GraphSpec

from typing import List, Dict, Any, Callable, Tuple

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Executable syntax tree to represent and calculate answers
# --------------------------------------------------------------------------

class FunctionalOperator:
    def __init__(self, *args):
        self.args: Tuple[Any, ...] = args

    def __call__(self, graph: GraphSpec) -> Any:
        """Execute this whole program to get an answer"""

        def ex(item):
            if isinstance(item, FunctionalOperator):
                return item(graph)
            # Handle NodeSpec/ arguments directly if they appear
            elif isinstance(item, NodeSpec):
                return item
            else:
                return item

        vals = [ex(i) for i in self.args]
        try:
            # Use the graph's networkx representation if needed by op
            return self.op(graph, *vals)
        except nx.NetworkXException as path_ex:
            # Specific handling for path errors which might be valid "no path" answers
            # Re-raise standard ValueErrors for generation logic, catch others
            logger.debug(f"NetworkX error in {type(self).__name__}({vals}): {path_ex}")
            raise ValueError(f"NetworkX error: {path_ex}") # Treat as generation failure
        except ValueError as ve:
            # Propagate ValueErrors used for control flow (e.g., ambiguous mode)
             logger.debug(f"ValueError in {type(self).__name__}({vals}): {ve}")
             raise ve
        except Exception as ex:
            logger.error(f"Failed to execute operation {type(self).__name__}({vals}): {ex}", exc_info=True)
            raise RuntimeError(f"Execution error in {type(self).__name__}: {ex}") from ex


    def op(self, graph: GraphSpec, *args: Any) -> Any:
        """
        Perform this individual operation.
        Operations should raise ValueError if it is not possible to generate
        a valid answer (e.g., no unique mode, item not found), but no *error* has occurred.
        This exception will be caught by the generation loop.
        Use graph.gnx for networkx operations.
        """
        raise NotImplementedError()

    def _serialize_arg(self, item):
        """Helper to serialize arguments for the functional representation."""
        if isinstance(item, FunctionalOperator):
            return item.to_dict()
        elif isinstance(item, NodeSpec):
            # Represent Nodes/Lines by their ID or Name in functional programs
            return {"Node" if isinstance(item, NodeSpec) else "Line": item.id}
        elif isinstance(item, type): # e.g. Node, Line classes themselves
             return item.__name__
        elif callable(item) and not isinstance(item, FunctionalOperator):
            # Handle lambdas - represent structure if possible
            try:
                sig = signature(item)
                # Create dummy args based on signature for representation
                dummy_args = [LambdaArg(p.name) for p in sig.parameters.values()]
                # Represent the body call with these dummy args
                body_repr = item(*dummy_args)
                if isinstance(body_repr, FunctionalOperator):
                     return {"Lambda": {"params": [p.name for p in sig.parameters.values()],
                                        "body": body_repr.to_dict()}}
                else:
                    # Cannot represent complex lambda body easily, return placeholder
                     return f"<lambda {sig}>"
            except (ValueError, TypeError): # Signature fails for some builtins/partials
                return f"<callable {item.__name__}>"

        # Check for basic types that are JSON serializable
        elif isinstance(item, (str, int, float, bool, list, dict, tuple)) or item is None:
             # Recursively serialize lists/tuples
             if isinstance(item, (list, tuple)):
                 return [self._serialize_arg(i) for i in item]
             # Let dicts pass through if simple, might need recursion if complex
             elif isinstance(item, dict) :
                 return {k: self._serialize_arg(v) for k, v in item.items()}
             return item
        else:
            # Fallback for unknown types
            return f"<unserializable: {type(item).__name__}>"


    def to_dict(self) -> Dict[str, Any]:
        """Represent this program as a serializable dictionary."""
        serialized_args = [self._serialize_arg(arg) for arg in self.args]
        return {type(self).__name__: serialized_args}

    def __repr__(self):
        # Basic representation, to_dict is better for structure
        arg_repr = ", ".join(map(repr, self.args))
        return f"{type(self).__name__}({arg_repr})"


def macro(f: Callable) -> Callable:
    # Keep the macro decorator, its usage seems fine
    return f

# --------------------------------------------------------------------------
#  Noun operations - Simplified, mainly used for type hinting/selection now
#  The actual values (NodeSpec, , str, etc.) are passed during generation.
# --------------------------------------------------------------------------
# Keep classes like Station, Line, Architecture etc. for QuestionForm definition
# But their 'get' methods are mainly illustrative now. The generator selects actual objects.

class Station(FunctionalOperator):
    # Represents a station argument. The actual NodeSpec is passed in.
    pass

class FakeStationName(FunctionalOperator):
    # Represents a fake name argument. The actual string is passed in.
     pass

class Line(FunctionalOperator):
    # Represents a line argument. The actual  is passed in.
    pass

# Add similar placeholders for Architecture, Size, Music, Cleanliness, Boolean if needed
# for type clarity in QuestionForm definitions.

class Architecture(FunctionalOperator): pass
class Size(FunctionalOperator): pass
class Music(FunctionalOperator): pass
class Cleanliness(FunctionalOperator): pass
class Boolean(FunctionalOperator): pass

# --------------------------------------------------------------------------
# General operations
# --------------------------------------------------------------------------

class Const(FunctionalOperator):
    def op(self, graph: GraphSpec, a):
        return a

# Lambda representation simplified in to_dict
class Lambda(FunctionalOperator):
    def op(self, graph: GraphSpec, func_repr): # Argument is the serialized representation
        # Cannot execute serialized lambdas directly. This is mainly for structure.
        raise NotImplementedError("Cannot execute Lambda representation directly.")

class LambdaArg(FunctionalOperator):
    def op(self, graph: GraphSpec, arg_name):
         # Placeholder for lambda arguments in functional representation
         return f"<arg:{arg_name}>"


class Pluck(FunctionalOperator):
    def op(self, graph: GraphSpec, collection: List[Any], key: str):
        # Handles lists of NodeSpec, EdgeSpec, , or dicts
        results = []
        for item in collection:
            try:
                if isinstance(item, (NodeSpec, EdgeSpec)):
                    results.append(item[key]) # Use __getitem__
                elif isinstance(item, dict):
                    results.append(item[key])
                else:
                     # Attempt getattr for other objects if necessary
                     results.append(getattr(item, key))
            except (KeyError, AttributeError):
                 # Handle cases where the key might not exist for an item
                 logger.warning(f"Key '{key}' not found in item {item} during Pluck.")
                 # Decide behaviour: skip, add None, or raise error? Let's skip.
                 # results.append(None) # Option to add None
                 pass # Option to skip
                 # raise ValueError(f"Key '{key}' not found in item {item}") # Option to fail
        return results


class Pick(FunctionalOperator):
    def op(self, graph: GraphSpec, item: Any, key: str):
        # Handles NodeSpec, EdgeSpec, , or dicts
        try:
            if isinstance(item, (NodeSpec, EdgeSpec)):
                return item[key] # Use __getitem__
            elif isinstance(item, dict):
                return item[key]
            else:
                 # Attempt getattr for other objects if necessary
                 return getattr(item, key)
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Cannot Pick key '{key}' from item {item}: {e}")


class Equal(FunctionalOperator):
    def op(self, graph: GraphSpec, a, b):
        # Special handling for comparing NodeSpec/ by ID
        if isinstance(a, (NodeSpec)) and isinstance(b, (NodeSpec)):
            return a.id == b.id
        return a == b

# --------------------------------------------------------------------------
# Graph operations - Use graph.gnx
# --------------------------------------------------------------------------

def ids_to_nodes(graph: GraphSpec, ids: List[str]) -> List[NodeSpec]:
    """Helper to convert node IDs back to NodeSpec objects."""
    return [graph.nodes[i] for i in ids if i in graph.nodes]

class AllEdges(FunctionalOperator):
    def op(self, graph: GraphSpec):
        return graph.edges # Return the list of EdgeSpec objects

class AllNodes(FunctionalOperator):
    def op(self, graph: GraphSpec):
        return list(graph.nodes.values()) # Return list of NodeSpec objects

class Edges(FunctionalOperator):
    def op(self, graph: GraphSpec, node_or_nodes: Any):
        target_node_ids = set()
        if isinstance(node_or_nodes, NodeSpec):
            target_node_ids.add(node_or_nodes.id)
        elif isinstance(node_or_nodes, list) and all(isinstance(n, NodeSpec) for n in node_or_nodes):
             target_node_ids.update(n.id for n in node_or_nodes)
        else:
             raise TypeError(f"Edges expects NodeSpec or List[NodeSpec], got {type(node_or_nodes)}")

        incident_edges = []
        # Iterate through original edges list for efficiency if graph large
        # or use gnx if already generated. Let's use gnx for simplicity here.
        try:
             # gnx stores EdgeSpec in edge data under 'data' key from our setup
            for node_id in target_node_ids:
                 if graph.gnx.has_node(node_id):
                    for u, v, edge_data in graph.gnx.edges(node_id, data=True):
                        if 'data' in edge_data and isinstance(edge_data['data'], EdgeSpec):
                             incident_edges.append(edge_data['data'])
                        else:
                             # Fallback if gnx wasn't populated with EdgeSpec correctly
                             # This requires searching self.edges - less efficient
                             logger.warning(f"Edge data for ({u},{v}) missing EdgeSpec, searching manually.")
                             found_edge = next((e for e in graph.edges if (e.station1 == u and e.station2 == v) or (e.station1 == v and e.station2 == u)), None)
                             if found_edge:
                                 incident_edges.append(found_edge)

            # Remove duplicates if a node was shared by multiple edges queried
            unique_edges = []
            seen_edge_keys = set()
            for edge in incident_edges:
                # Define uniqueness based on the two connected stations (order invariant) and the line
                key = tuple(sorted((edge.station1, edge.station2))) + (edge.line_id,)
                if key not in seen_edge_keys:
                    unique_edges.append(edge)
                    seen_edge_keys.add(key)
            return unique_edges
        except Exception as e:
            logger.error(f"Error accessing edges via gnx: {e}")
            # Fallback: Iterate through graph.edges (less efficient for many nodes)
            incident_edges = []
            for edge_spec in graph.edges:
                if edge_spec.station1 in target_node_ids or edge_spec.station2 in target_node_ids:
                    incident_edges.append(edge_spec)
            
            # Remove duplicates based on station pair and line ID
            unique_edges = []
            seen_edge_keys = set()
            for edge in incident_edges:
                # Define uniqueness based on the two connected stations (order invariant) and the line
                key = tuple(sorted((edge.station1, edge.station2))) + (edge.line_id,)
                if key not in seen_edge_keys:
                    unique_edges.append(edge)
                    seen_edge_keys.add(key)
            return unique_edges


class Nodes(FunctionalOperator):
     def op(self, graph: GraphSpec, edges: List[EdgeSpec]):
         node_ids = set()
         for edge in edges:
             node_ids.add(edge.station1)
             node_ids.add(edge.station2)
         # Convert unique IDs back to NodeSpec objects
         return [graph.nodes[nid] for nid in node_ids if nid in graph.nodes]


class ShortestPath(FunctionalOperator):
    def op(self, graph: GraphSpec, a: NodeSpec, b: NodeSpec, fallback: Any):
        if not isinstance(a, NodeSpec) or not isinstance(b, NodeSpec):
             raise TypeError(f"ShortestPath expects NodeSpec arguments, got {type(a)}, {type(b)}")
        try:
            path_ids = nx.shortest_path(graph.gnx, a.id, b.id)
            return ids_to_nodes(graph, path_ids)
        except nx.NetworkXNoPath:
            logger.debug(f"No path between {a.id} and {b.id}")
            return fallback # Return the specified fallback value
        except nx.NodeNotFound:
             logger.warning(f"Node not found in shortest path calculation: {a.id} or {b.id}")
             # Reraise as ValueError to indicate generation issue
             raise ValueError(f"Node not found for shortest path: {a.id} or {b.id}")


class ShortestPathOnlyUsing(FunctionalOperator):
     def op(self, graph: GraphSpec, a: NodeSpec, b: NodeSpec, only_using_nodes: List[NodeSpec], fallback: Any):
         if not isinstance(a, NodeSpec) or not isinstance(b, NodeSpec):
              raise TypeError(f"ShortestPathOnlyUsing expects NodeSpec args, got {type(a)}, {type(b)}")
         if not isinstance(only_using_nodes, list) or not all(isinstance(n, NodeSpec) for n in only_using_nodes):
             raise TypeError(f"ShortestPathOnlyUsing expects List[NodeSpec] for only_using_nodes")

         allowed_node_ids = {node.id for node in only_using_nodes} | {a.id, b.id}
         subgraph_nodes = [nid for nid in allowed_node_ids if graph.gnx.has_node(nid)]

         if a.id not in subgraph_nodes or b.id not in subgraph_nodes:
             logger.debug(f"Start/end node {a.id}/{b.id} not in allowed subgraph for ShortestPathOnlyUsing")
             return fallback # Start or end node not even allowed

         subgraph = graph.gnx.subgraph(subgraph_nodes)
         try:
             path_ids = nx.shortest_path(subgraph, a.id, b.id)
             return ids_to_nodes(graph, path_ids)
         except nx.NetworkXNoPath:
             logger.debug(f"No path between {a.id} and {b.id} using only allowed nodes.")
             return fallback
         except nx.NodeNotFound:
              # This shouldn't happen if we check subgraph_nodes inclusion, but defensively handle.
              logger.warning(f"Node not found in subgraph shortest path: {a.id} or {b.id}")
              raise ValueError(f"Node not found in subgraph shortest path: {a.id} or {b.id}")


class Paths(FunctionalOperator):
    def op(self, graph: GraphSpec, a: NodeSpec, b: NodeSpec):
        if not isinstance(a, NodeSpec) or not isinstance(b, NodeSpec):
             raise TypeError(f"Paths expects NodeSpec arguments, got {type(a)}, {type(b)}")
        try:
             if not graph.gnx.has_node(a.id) or not graph.gnx.has_node(b.id):
                 raise ValueError(f"Node not found for Paths: {a.id} or {b.id}")
             all_paths_ids = list(nx.all_simple_paths(graph.gnx, a.id, b.id))
             return [ids_to_nodes(graph, p) for p in all_paths_ids]
        except nx.NodeNotFound:
             logger.warning(f"Node not found for Paths: {a.id} or {b.id}")
             raise ValueError(f"Node not found for Paths: {a.id} or {b.id}")


class HasCycle(FunctionalOperator):
     def op(self, graph: GraphSpec, start_node: NodeSpec):
         if not isinstance(start_node, NodeSpec):
             raise TypeError(f"HasCycle expects a NodeSpec argument, got {type(start_node)}")
         if not graph.gnx.has_node(start_node.id):
             raise ValueError(f"Node {start_node.id} not in graph for HasCycle check.")

         try:
             # find_cycle is efficient for finding *a* cycle involving the node (or any if node is None)
             # Check cycles involving the specific node. Edges from find_cycle need careful handling.
             # Let's use a simpler approach: check if the node is part of any cycle component.
             cycles = list(nx.cycle_basis(graph.gnx)) # Find all fundamental cycles
             start_node_id = start_node.id
             for cycle in cycles:
                 if start_node_id in cycle:
                     return True
             return False
             # Alternative: DFS approach from original code (might be slow)
             # The original DFS implementation looks complex and potentially inefficient.
             # Relying on networkx's cycle finding is safer.
         except Exception as e:
              logger.error(f"Error during cycle detection for node {start_node.id}: {e}")
              # Decide on fallback behavior, maybe assume no cycle or raise error
              raise ValueError(f"Cycle detection failed for {start_node.id}") from e


class FilterAdjacent(FunctionalOperator):
    def op(self, graph: GraphSpec, list_a: List[NodeSpec], list_b: List[NodeSpec]):
         if not isinstance(list_a, list) or not all(isinstance(n, NodeSpec) for n in list_a):
             raise TypeError(f"FilterAdjacent expects List[NodeSpec] for list_a")
         if not isinstance(list_b, list) or not all(isinstance(n, NodeSpec) for n in list_b):
             raise TypeError(f"FilterAdjacent expects List[NodeSpec] for list_b")

         results = []
         # Pre-fetch neighbors for efficiency if lists are large
         neighbors_b = {node_b.id: set(graph.gnx.neighbors(node_b.id)) for node_b in list_b if graph.gnx.has_node(node_b.id)}

         for node_a in list_a:
            if not graph.gnx.has_node(node_a.id): continue
            neighbors_a = set(graph.gnx.neighbors(node_a.id))
            for node_b in list_b:
                if node_a.id == node_b.id: continue # Don't pair with self
                if node_b.id in neighbors_a: # Check if b is a neighbor of a
                     # Original code returned [node_a, node_b]. Confirm requirement.
                     results.append([node_a, node_b])
                     # If only node_a is needed: results.append(node_a); break inner loop?
         return results # Returns list of pairs [a, b] where a is from list_a, b from list_b, and they are adjacent


class Neighbors(FunctionalOperator):
    def op(self, graph: GraphSpec, station: NodeSpec):
        if not isinstance(station, NodeSpec):
             raise TypeError(f"Neighbors expects a NodeSpec argument, got {type(station)}")
        if not graph.gnx.has_node(station.id):
             raise ValueError(f"Node {station.id} not in graph for Neighbors.")
        neighbor_ids = list(graph.gnx.neighbors(station.id))
        return ids_to_nodes(graph, neighbor_ids)


class WithinHops(FunctionalOperator):
    def op(self, graph: GraphSpec, station: NodeSpec, hops: int):
        if not isinstance(station, NodeSpec):
             raise TypeError(f"WithinHops expects a NodeSpec start station, got {type(station)}")
        if not isinstance(hops, int) or hops < 0:
            raise TypeError(f"WithinHops expects a non-negative integer for hops, got {hops}")
        if not graph.gnx.has_node(station.id):
            raise ValueError(f"Node {station.id} not in graph for WithinHops.")

        if hops == 0:
            return [] # No nodes within 0 hops (excluding start node)

        # Use networkx's single_source_shortest_path_length
        try:
            distances = nx.single_source_shortest_path_length(graph.gnx, station.id, cutoff=hops)
            # Result includes the start node with distance 0. Exclude it.
            nearby_node_ids = [node_id for node_id, dist in distances.items() if node_id != station.id and dist <= hops]
            return ids_to_nodes(graph, nearby_node_ids)
        except nx.NodeNotFound:
             # Should be caught by the has_node check, but handle defensively
             raise ValueError(f"Node {station.id} not found during WithinHops distance calculation.")


class FilterHasPathTo(FunctionalOperator):
     def op(self, graph: GraphSpec, node_list: List[NodeSpec], target_node: NodeSpec):
          if not isinstance(node_list, list) or not all(isinstance(n, NodeSpec) for n in node_list):
              raise TypeError(f"FilterHasPathTo expects List[NodeSpec] for node_list")
          if not isinstance(target_node, NodeSpec):
              raise TypeError(f"FilterHasPathTo expects NodeSpec for target_node")
          if not graph.gnx.has_node(target_node.id):
              raise ValueError(f"Target node {target_node.id} not in graph for FilterHasPathTo.")

          results = []
          for source_node in node_list:
              if source_node.id == target_node.id: # Path to self exists
                  results.append(source_node)
                  continue
              if not graph.gnx.has_node(source_node.id):
                   logger.warning(f"Source node {source_node.id} in list but not in graph for FilterHasPathTo.")
                   continue
              try:
                  if nx.has_path(graph.gnx, source_node.id, target_node.id):
                      results.append(source_node)
              except nx.NodeNotFound:
                   # Should be caught by checks, but handle defensively
                   logger.warning(f"Node not found during nx.has_path check: {source_node.id} or {target_node.id}")
                   continue
          return results

# --------------------------------------------------------------------------
# List operators
# --------------------------------------------------------------------------

class NotEmpty(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        return len(l) > 0

class Count(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        return len(l)

class CountIfEqual(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List, t: Any):
        # Handle comparison with NodeSpec/ using ID
        count = 0
        for item in l:
            if isinstance(item, (NodeSpec, )) and isinstance(t, (NodeSpec, )):
                if item.id == t.id:
                    count += 1
            elif item == t:
                 count += 1
        return count

class CountIfIn(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List, tl: List):
        # Handle comparison with NodeSpec/ using ID
        count = 0
        for item in l:
            for t in tl:
              if isinstance(item, (NodeSpec, )) and isinstance(t, (NodeSpec, )):
                  if item.id == t.id:
                      count += 1
                      break
              elif item == t:
                  count += 1
                  break
        return count


class Mode(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        if not l:
            raise ValueError("Cannot find mode of empty sequence")

        # Handle unhashable types like NodeSpec/ by using their ID
        processed_list = []
        for item in l:
            if isinstance(item, (NodeSpec, )):
                processed_list.append(item.id) # Use ID for counting
            elif isinstance(item, list): # Cannot count lists directly
                 raise ValueError("Mode cannot operate on lists containing lists.")
            # Add handling for other unhashable types if needed
            else:
                processed_list.append(item)

        c = Counter(processed_list)
        most = c.most_common(2)

        if len(most) == 1 or most[0][1] > most[1][1]:
             # If unique mode found based on ID, need to return the original object
             mode_val = most[0][0]
             # Find the first original item that corresponds to the mode value/ID
             for item in l:
                 if isinstance(item, (NodeSpec, )) and item.id == mode_val:
                     return item
                 elif not isinstance(item, (NodeSpec, )) and item == mode_val:
                     return item
             # Should not happen if mode_val came from the list
             raise RuntimeError("Could not find original item for calculated mode.")
        else:
            raise ValueError("No unique mode")


class Unique(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        # Handle NodeSpec/ based on ID for uniqueness
        seen_ids = set()
        unique_list = []
        # Preserve order while ensuring uniqueness
        for item in l:
            if isinstance(item, (NodeSpec, )):
                if item.id not in seen_ids:
                    unique_list.append(item)
                    seen_ids.add(item.id)
            # Handle other hashable types
            elif item not in unique_list: # This check is O(n) for list, could use set if hashable
                 try:
                     if item not in seen_ids: # Assume hashable, use set for speed
                          unique_list.append(item)
                          seen_ids.add(item)
                 except TypeError: # Handle unhashable types (e.g. lists) linearly
                      if item not in unique_list:
                           unique_list.append(item)
        return unique_list


class SlidingPairs(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        if len(l) < 2:
            return []
        return [(l[i], l[i+1]) for i in range(len(l)-1)]


@macro
def GetLines(a: Any) -> FunctionalOperator: # a can be NodeSpec or List[NodeSpec]
    # Needs refinement: Edges(a) returns EdgeSpecs. Pluck needs 'line_name' or maybe ?
    # Let's return  objects for consistency
    # Pluck(Edges(a), "line_id") -> get LineSpecs from graph.lines using these IDs
    # This macro needs to be implemented carefully or avoided if too complex
    # Simplified version returning line names:
    return Unique(Pluck(Edges(a), "line_name"))
    # Better version returning  objects:
    # 1. Get Edges -> List[EdgeSpec]
    # 2. Pluck "line_id" -> List[str]
    # 3. Unique -> List[str]
    # 4. Map ID to  -> List[] - This requires a new operator MapLineIdToLineSpec
    # For now, stick to the simpler line name version.

@macro
def Adjacent(a: NodeSpec, b: NodeSpec) -> FunctionalOperator:
    # Check if distance is 1. ShortestPath returns list of nodes.
    # If len(ShortestPath) == 2, they are adjacent.
    # Need robust fallback handling in ShortestPath
    # path = ShortestPath(a, b, []) # Fallback empty list if no path
    # return Equal(Count(path), 2)
    # More direct: check gnx.neighbors
    return IsNeighbor(a, b) # Requires a new IsNeighbor operator

# Let's define IsNeighbor explicitly
class IsNeighbor(FunctionalOperator):
    def op(self, graph: GraphSpec, a: NodeSpec, b: NodeSpec):
        if not isinstance(a, NodeSpec) or not isinstance(b, NodeSpec):
             raise TypeError("IsNeighbor expects NodeSpec arguments")
        if not graph.gnx.has_node(a.id) or not graph.gnx.has_node(b.id):
             return False # Node not even in graph
        return graph.gnx.has_edge(a.id, b.id)

# Redefine Adjacent macro using IsNeighbor
@macro
def Adjacent(a: NodeSpec, b: NodeSpec) -> FunctionalOperator:
    return IsNeighbor(a,b)


@macro
def CountNodesBetween(path: List[NodeSpec]) -> FunctionalOperator:
     # Path includes start and end, so subtract 2
     # Ensure path is a list (result of ShortestPath)
     # This macro takes the *result* of ShortestPath, not the operation itself
     # It should be applied like: Subtract(Count(ShortestPath(a, b)), 2)
     # So, the macro definition itself is tricky. Better to write it out directly
     # where needed, e.g., lambda path: Subtract(Count(path), 2)
     # Let's remove this macro and use Subtract(Count(ShortestPath(...)), 2) directly.
     pass # Remove this macro, use direct composition

class HasIntersection(FunctionalOperator):
    def op(self, graph: GraphSpec, list_a: List, list_b: List):
         # Use sets for efficient intersection checking, handle NodeSpec/ IDs
         set_a = set()
         for item in list_a:
             if isinstance(item, (NodeSpec, )):
                 set_a.add(item.id)
             else: # Assume hashable
                 try: set_a.add(item)
                 except TypeError: # Handle unhashable comparison linearly
                      if item in list_b: return True

         set_b = set()
         for item in list_b:
             if isinstance(item, (NodeSpec, )):
                 set_b.add(item.id)
             else: # Assume hashable
                 try: set_b.add(item)
                 except TypeError: # Already handled in linear check above
                      pass

         # Efficient check for hashable intersection
         if not set_a.isdisjoint(set_b):
             return True

         # Linear check for any remaining unhashable items in list_a vs list_b
         # (Only needed if the first linear check didn't find anything and set intersect was false)
         unhashable_a = [item for item in list_a if isinstance(item, (list, dict))] # Example unhashables
         if unhashable_a:
             unhashable_b = [item for item in list_b if isinstance(item, (list, dict))]
             for item_a in unhashable_a:
                 if item_a in unhashable_b:
                     return True

         return False # No intersection found


class Intersection(FunctionalOperator):
    def op(self, graph: GraphSpec, list_a: List, list_b: List):
         # Similar logic to HasIntersection, but return the common items
         # This is complex with mixed types and NodeSpec/ IDs.
         # Simplification: Assume inputs are primarily NodeSpec/ or simple hashables.
         # Return original objects corresponding to intersecting IDs/values.

         map_a = {} # Store original items by hashable representation (ID or value)
         processed_a = set()
         for item in list_a:
             if isinstance(item, (NodeSpec, )):
                 key = item.id
                 if key not in map_a: map_a[key] = item
                 processed_a.add(key)
             else: # Assume hashable
                  try:
                      key = item
                      if key not in map_a: map_a[key] = item
                      processed_a.add(key)
                  except TypeError: pass # Skip unhashable for set intersection part

         map_b = {}
         processed_b = set()
         for item in list_b:
             if isinstance(item, (NodeSpec, )):
                 key = item.id
                 if key not in map_b: map_b[key] = item
                 processed_b.add(key)
             else: # Assume hashable
                 try:
                     key = item
                     if key not in map_b: map_b[key] = item
                     processed_b.add(key)
                 except TypeError: pass

         common_keys = processed_a.intersection(processed_b)
         # Retrieve original items from list_a corresponding to common keys
         # Preserve order from list_a where possible
         intersection_result = [map_a[key] for key in map_a if key in common_keys]

         # Add linear check for unhashables if necessary, similar to HasIntersection
         # ...

         return intersection_result


class Filter(FunctionalOperator):
     def op(self, graph: GraphSpec, collection: List[Any], key: str, value: Any):
         results = []
         for item in collection:
             try:
                 item_value = None
                 if isinstance(item, (NodeSpec, EdgeSpec, )):
                     item_value = item[key] # Use __getitem__
                 elif isinstance(item, dict):
                     item_value = item.get(key)
                 else:
                     item_value = getattr(item, key, None)

                 # Compare item_value with target value
                 # Handle NodeSpec/ comparison by ID if needed
                 should_include = False
                 if isinstance(item_value, (NodeSpec, )) and isinstance(value, (NodeSpec, )):
                     if item_value.id == value.id:
                         should_include = True
                 elif item_value == value:
                     should_include = True

                 if should_include:
                     results.append(item)

             except (KeyError, AttributeError):
                  # Key doesn't exist for this item, skip it
                  pass
             except Exception as e:
                  logger.error(f"Error during Filter comparison for item {item}: {e}")
                  pass # Skip item on error
         return results


class Without(FunctionalOperator):
     def op(self, graph: GraphSpec, collection: List[Any], key: str, value: Any):
         # Similar logic to Filter, but include if the value *doesn't* match
         results = []
         for item in collection:
             try:
                 item_value = None
                 if isinstance(item, (NodeSpec, EdgeSpec, )):
                     item_value = item[key]
                 elif isinstance(item, dict):
                     item_value = item.get(key)
                 else:
                     item_value = getattr(item, key, None)

                 # Compare item_value with target value
                 matches = False
                 if isinstance(item_value, (NodeSpec, )) and isinstance(value, (NodeSpec, )):
                     if item_value.id == value.id:
                         matches = True
                 elif item_value == value:
                      matches = True

                 if not matches: # Include if it *doesn't* match
                     results.append(item)

             except (KeyError, AttributeError):
                  # Key doesn't exist, so it doesn't match the value - include it
                  results.append(item)
             except Exception as e:
                 logger.error(f"Error during Without comparison for item {item}: {e}")
                 results.append(item) # Include item on error? Or skip? Let's include.
         return results

class UnpackUnitList(FunctionalOperator):
    def op(self, graph: GraphSpec, l: List):
        if len(l) == 1:
            return l[0]
        else:
            # This should be caught by the generator to avoid ambiguous questions
            raise ValueError(f"List is length {len(l)}, expected 1 for UnpackUnitList")


class Sample(FunctionalOperator):
     def op(self, graph: GraphSpec, l: List, n: int):
         if not isinstance(n, int) or n < 0:
             raise ValueError(f"Sample requires a non-negative integer n, got {n}")
         if len(l) < n:
             # This should be caught by the generator if sampling without replacement is implied
             raise ValueError(f"Cannot sample {n} items from list of length {len(l)}")
         if n == 0:
             return []
         # random.sample does sampling without replacement
         # return random.sample(l, k=n)
         # random.choices does sampling *with* replacement
         return random.choices(l, k=n) # Use choices as per original code


class First(FunctionalOperator):
     def op(self, graph: GraphSpec, l: List):
         if not l:
             raise ValueError("Cannot get First element of an empty list")
         return l[0]


class MinBy(FunctionalOperator):
     def op(self, graph: GraphSpec, collection: List[Any], key_func: FunctionalOperator):
         if not collection:
             raise ValueError("Cannot perform MinBy on empty list")
         if not isinstance(key_func, FunctionalOperator):
              # Allow simple functions too? For now, assume FunctionalOperator representation
              raise TypeError("MinBy expects a FunctionalOperator as key_func")

         min_item = None
         min_value = float('inf')

         for item in collection:
             try:
                 # The key_func needs to be callable and take the item.
                 # This structure is complex. Assume key_func is like Lambda(lambda x: Count(ShortestPath(start_node, x)))
                 # We need to execute the key_func *with the current item bound*.
                 # This requires modifying the key_func structure or execution.

                 # Simplification: Assume key_func is a simple lambda stored somehow, or a string key.
                 # If key_func represents a lambda like `lambda y: Count(ShortestPath(x, y))`
                 # where `x` is fixed and `y` is the item from `collection`.
                 # Let's assume the key_func structure allows binding 'item'.
                 # This part of the original design might need a rethink for clean execution.

                 # --- TEMPORARY WORKAROUND ---
                 # Assume key_func is a simple lambda passed directly (won't work with serialization)
                 # Or that the functional structure handles binding correctly (complex).
                 # Let's simulate the intended behaviour assuming key_func was a python lambda:
                 if callable(key_func): # If key_func *was* a python lambda (won't be after serialization)
                      value = key_func(item) # THIS WON'T WORK WITH FunctionalOperator structure
                 else:
                      # Try evaluating the key_func structure - needs careful implementation
                      # This is hard to do generally without full evaluation context.
                      # For now, let's raise an error indicating this complexity.
                      raise NotImplementedError("MinBy with complex FunctionalOperator key_func execution is not fully implemented.")

                 # --- END WORKAROUND ---

                 # value = key_func(item)(graph) # Idealized execution

                 if value < min_value:
                     min_value = value
                     min_item = item
             except Exception as e:
                 logger.error(f"Error evaluating MinBy key function for item {item}: {e}")
                 # Skip item or raise error? Skip for now.
                 continue

         if min_item is None:
             raise ValueError("Could not determine minimum value in MinBy (all key functions may have failed)")
         return min_item


# --------------------------------------------------------------------------
# Numerical operations
# --------------------------------------------------------------------------

class Subtract(FunctionalOperator):
    def op(self, graph: GraphSpec, a, b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            # Allow subtraction from list length, etc. Ensure 'a' becomes numeric first.
            if isinstance(a, list): a = len(a)
            if isinstance(b, list): b = len(b)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                raise TypeError(f"Subtract requires numerical arguments, got {type(a)}, {type(b)}")
        return a - b

class Round(FunctionalOperator):
     def op(self, graph: GraphSpec, a):
         try:
             if isinstance(a, list):
                 # Attempt to round elements if they are numeric
                 return [round(float(i)) if isinstance(i, (int, float, str)) and str(i).isdigit() else i for i in a]
             else:
                 return round(float(a)) # Attempt to convert and round
         except (ValueError, TypeError) as e:
              raise TypeError(f"Cannot Round value {a} ({type(a)}): {e}")
          
class Max(FunctionalOperator):
    def op(self, graph: GraphSpec, a, b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            # Allow max with list length
            if isinstance(a, list): a = len(a)
            if isinstance(b, list): b = len(b)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                raise TypeError(f"Max requires numerical arguments, got {type(a)}, {type(b)}")
        return max(a, b)