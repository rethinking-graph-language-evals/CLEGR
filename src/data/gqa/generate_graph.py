import uuid
import random
import numpy as np
from sklearn.neighbors import KDTree
import bezier
from gibberish import Gibberish
import logging
from collections import Counter, defaultdict
import networkx as nx

from typing import List, Dict, Set, Optional, Any

logger = logging.getLogger(__name__)

from .types_ import GraphSpec, NodeSpec, EdgeSpec, LineSpec
from .args import *

gib = Gibberish()

# --- Constants for Generation ---
# (Keep these property dictionaries as they are used for random selection)
LineProperties = {
    "has_aircon": [True, False],
    "color": ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'olive', 'cyan'],
    "stroke": ["solid" , "dashed", "dashdot", "dotted"],
    "built": ["1990", "2000", "2010", "2020", "2025", "1980"],
}

StationProperties = {
    "disabled_access": [True, False],
    "has_rail": [True, False],
    "music": ["classical", "rock", "electronic", "country", "none", "swing", "pop"],
    "architecture": ["victorian", "modernist", "concrete", "glass", "art-deco", "new"],
    "size": ["tiny", "small", "medium-sized", "large"],
    "cleanliness": ["clean", 'dirty'],
}

OtherProperties = {
    "surname": [" street", " st", " road", " court", " grove", "bridge", " bridge", " lane", " way", " boulevard", " crossing", " square", "ham", ' on trent', ' upon Thames', ' international', ' hospital', 'neyland', 'ington', 'ton', 'wich', ' manor', ' estate', ' palace']
}

# --- Helper Functions ---
def gen_n(base, noise = 0.2):
    """Generates a number based on base value with Gaussian noise."""
    # Ensure non-negative result, minimum 1?
    return max(1, round(random.gauss(base, noise*base)))

def add_noise(base, noise=0.05):
    """Adds uniform noise to a base value."""
    return base * (1 - noise + random.random() * noise*2)

# --- Graph Generator Class ---
class GraphGenerator:

    def __init__(self, args):
        self.args = args

        # Default statistics
        self.stats = {
            "lines": 22,
            "stations_per_line": 20,
            "map_radius": 25,
            "min_station_dist": 0.8,
            "coalesce_iterations": 15, # Limit coalesce steps
        }

        # Adjust stats based on args
        if args.medium:
            logger.info("Using tiny graph settings.")
            self.stats["lines"] = 10
            self.stats["stations_per_line"] = 10
            self.stats["map_radius"] = 15
            self.stats["min_station_dist"] = 0.85
            self.stats["coalesce_iterations"] = 12
        elif args.small:
            logger.info("Using small graph settings.")
            self.stats["lines"] = 6
            self.stats["stations_per_line"] = 7
            self.stats["map_radius"] = 8
            self.stats["min_station_dist"] = 0.9
            self.stats["coalesce_iterations"] = 10
        elif args.mixed:
            logger.info("Using mixed graph settings.")
            self.stats["lines"] = [22, 12, 6]
            self.stats["stations_per_line"] = [20, 12, 7]
            self.stats["map_radius"] = [25, 15, 8]
            self.stats["min_station_dist"] = [1, 0.85, 0.9]
            self.stats["coalesce_iterations"] = [15, 12, 10]

        # Internal state during generation
        self.line_set: List[LineSpec] = []
        self.station_gen_set: Set[NodeSpec] = set() # Track generated stations by ID/name hash
        self.line_stations: Dict[str, List[NodeSpec]] = {} # line.id -> List[NodeSpec]
        self.current_nodes: Dict[str, NodeSpec] = {} # All unique nodes after coalesce: node.id -> NodeSpec
        self.current_edges: List[EdgeSpec] = []
        self.graph_spec: Optional[GraphSpec] = None

    def _generate_id(self):
         """Generates a unique ID."""
         # Using UUID for robust uniqueness, can switch to simpler if needed
         return str(uuid.uuid4())
         # Alternative: simpler integer IDs if preferred, requires careful tracking
         # self._next_id = getattr(self, '_next_id', 0) + 1
         # return str(self._next_id)


    def _generate_name(self, prefix: str, suffix_pool: List[str]) -> str:
        """Generates a plausible name."""
        name = prefix + random.choice(suffix_pool)
        return name.title() # Capitalize words

    def gen_line(self) -> LineSpec:
        """Generates a single LineSpec with random properties."""
        line_id = self._generate_id()
        props = {key: random.choice(values) for key, values in LineProperties.items()}
        name = self._generate_name(props["color"], [" Line", " Rail", " Way", " Express", " Tube"]) # Example suffixes
        # Ensure name uniqueness? Less critical for lines maybe.
        return LineSpec(id=line_id, name=name, **props)

    def gen_station(self) -> NodeSpec:
        """Generates a single NodeSpec with random properties."""
        station_id = self._generate_id()
        props = {key: random.choice(values) for key, values in StationProperties.items()}
        name = self._generate_name(gib.generate_word(), OtherProperties["surname"])

        # Add dummy coordinates, will be set properly later
        return NodeSpec(id=station_id, name=name, x=0.0, y=0.0, **props)

    def _get_unique_station(self) -> NodeSpec:
        """Generates a station, ensuring its name is unique within this generation."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            station = self.gen_station()
            # Check name uniqueness based on current generated set
            # Using NodeSpec's __hash__ which should be based on ID now
            is_unique = True
            for existing_station in self.station_gen_set:
                 if existing_station.name == station.name:
                      is_unique = False
                      break
            if is_unique:
                 self.station_gen_set.add(station) # Add to set tracking generated stations
                 return station
            attempts += 1
        raise RuntimeError(f"Failed to generate a unique station name after {max_attempts} attempts.")

    def gen_lines(self, size_id=None):
        """Generates the set of lines for the graph."""
        self.line_set = []
        if isinstance(self.stats["lines"], list):
            target_lines = gen_n(self.stats["lines"][size_id])
        else:
            target_lines = gen_n(self.stats["lines"])
        generated_line_names = set()
        generated_line_colors_strokes = set() # Avoid identical color/stroke pairs

        while len(self.line_set) < target_lines:
             line = self.gen_line()
             color_stroke = (line.color, line.stroke)
             # Check for uniqueness of name and color/stroke combination
             if line.name not in generated_line_names and color_stroke not in generated_line_colors_strokes:
                 self.line_set.append(line)
                 generated_line_names.add(line.name)
                 generated_line_colors_strokes.add(color_stroke)
             # Add retry limit?
        logger.debug(f"Generated {len(self.line_set)} unique lines.")


    def gen_stations_along_lines(self, size_id=None):
        """Generates stations positioned along bezier curves for each line."""
        self.station_gen_set = set() # Reset stations for this graph
        self.line_stations = {} # line.id -> List[NodeSpec] (initially uncoalesced)

        if isinstance(self.stats["map_radius"], list):
            map_radius = self.stats["map_radius"][size_id]
        else:
            map_radius = self.stats["map_radius"]

        for line in self.line_set:
            # Define bezier curve control points within map bounds
            xs = [(random.random() * 2 - 1) * map_radius for _ in range(4)]
            ys = [(random.random() * 2 - 1) * map_radius for _ in range(4)]
            # Ensure control points are somewhat spread out?
            nodes = np.asfortranarray([xs, ys])
            curve = bezier.Curve(nodes, degree=3) # Degree 3 for 4 control points
            if isinstance(self.stats["stations_per_line"], list):
                num_stations = max(2, gen_n(self.stats["stations_per_line"][size_id]))
            else:
                num_stations = max(2, gen_n(self.stats["stations_per_line"]))
            # Evaluate points along the curve
            s_vals = np.linspace(0.0, 1.0, num_stations)
            points = curve.evaluate_multi(s_vals) # Shape (2, num_stations)

            line_station_list: List[NodeSpec] = []
            for i in range(points.shape[1]):
                x, y = points[:, i]
                station = self._get_unique_station() # Get a unique station

                # Apply noise to position
                station.x = float(add_noise(x))
                station.y = float(add_noise(y))

                line_station_list.append(station)

            # Check for duplicate stations *within the same line* before storing
            # (Should be rare due to _get_unique_station, but coalesce handles inter-line later)
            self.line_stations[line.id] = line_station_list

        logger.debug(f"Generated initial stations for {len(self.line_stations)} lines.")
        logger.debug(f"Total unique stations generated initially: {len(self.station_gen_set)}")


    def coalesce_nearby_stations(self, size_id=None):
         """Merges stations that are physically close to each other."""
         if isinstance(self.stats["min_station_dist"], list):
            min_dist = self.stats["min_station_dist"][size_id]
         else:
            min_dist = self.stats["min_station_dist"]
         if min_dist <= 0:
              logger.debug("Skipping station coalescing (min_dist <= 0).")
              return

         logger.debug(f"Starting station coalescing (min_dist={min_dist})...")

         # Use the set of all unique stations generated
         all_stations_list = list(self.station_gen_set)
         if len(all_stations_list) < 2:
             logger.debug("Not enough stations to coalesce.")
             return # Need at least 2 stations

         # Keep track of which station ID maps to which 'master' station ID after merging
         # Initialize mapping: each station maps to itself
         station_map: Dict[str, str] = {s.id: s.id for s in all_stations_list}
         # Store the representative NodeSpec for each master ID
         master_nodes: Dict[str, NodeSpec] = {s.id: s for s in all_stations_list}

         coalesced_in_iteration = True
         iterations = 0
         if isinstance(self.stats["coalesce_iterations"], list):
             max_iterations = self.stats["coalesce_iterations"][size_id]
         else:
             max_iterations = self.stats["coalesce_iterations"]

         while coalesced_in_iteration and iterations < max_iterations:
             coalesced_in_iteration = False
             iterations += 1
             logger.debug(f"Coalesce iteration {iterations}")

             # Prepare data for KDTree: points and corresponding current master IDs
             current_master_ids = list(master_nodes.keys())
             if len(current_master_ids) < 2: break # Stop if only one cluster left

             points = np.array([[master_nodes[mid].x, master_nodes[mid].y] for mid in current_master_ids])
             master_id_map = {i: mid for i, mid in enumerate(current_master_ids)} # KDTree index -> master_id

             try:
                 # Build KDTree on current master node positions
                 tree = KDTree(points, leaf_size=10) # Adjust leaf_size if needed
                 # Query for pairs within min_dist
                 # query_pairs is efficient for finding all pairs within a distance
                 nearby_pairs_indices = tree.query_radius(points, r=min_dist)
                 # nearby_pairs_indices[i] contains indices of points near points[i]

             except ValueError as e:
                 logger.error(f"KDTree error during coalescing (possible issue with points array?): {e}")
                 logger.error(f"Points array shape: {points.shape}, first few points: {points[:3]}")
                 break # Stop coalescing on error


             # --- Process nearby pairs to merge clusters ---
             # Use a Disjoint Set Union (DSU) structure for efficient merging
             parent = {mid: mid for mid in current_master_ids}
             def find_set(v):
                 if v == parent[v]: return v
                 parent[v] = find_set(parent[v]) # Path compression
                 return parent[v]

             def unite_sets(a, b):
                 a_root = find_set(a)
                 b_root = find_set(b)
                 if a_root != b_root:
                     # Merge: keep the one with the 'smaller' ID (arbitrary but consistent)
                     if a_root < b_root:
                         parent[b_root] = a_root
                     else:
                         parent[a_root] = b_root
                     return True # Indicates a merge happened
                 return False

             merges_done = 0
             processed_pairs = set() # Avoid processing (i, j) and (j, i) separately

             for i, neighbors in enumerate(nearby_pairs_indices):
                 master_id_i = master_id_map[i]
                 for j in neighbors:
                     if i == j: continue # Don't pair with self
                     pair = tuple(sorted((i, j)))
                     if pair in processed_pairs: continue # Already processed this pair
                     processed_pairs.add(pair)

                     master_id_j = master_id_map[j]

                     # Perform union using DSU
                     if unite_sets(master_id_i, master_id_j):
                         merges_done += 1
                         coalesced_in_iteration = True # Mark that work was done

             if not coalesced_in_iteration:
                 logger.debug("No further merges in this iteration.")
                 break # Exit loop if no merges occurred

             logger.debug(f"Iteration {iterations}: {merges_done} merges performed.")

             # --- Update master nodes and station_map based on DSU results ---
             new_master_nodes = {}
             temp_station_map = station_map.copy() # Work on a copy

             # Find the root (representative) for each original station
             for original_station_id in list(station_map.keys()): # Iterate original IDs
                 current_master = temp_station_map[original_station_id]
                 root_master = find_set(current_master) # Find the ultimate root

                 # Update the station_map: original station now maps to the root master
                 station_map[original_station_id] = root_master

                 # Collect the root masters and their corresponding NodeSpec
                 if root_master not in new_master_nodes:
                     # Use the NodeSpec of the root master itself
                     new_master_nodes[root_master] = master_nodes[root_master]

             # Update master_nodes for the next iteration
             master_nodes = new_master_nodes
             logger.debug(f"Iteration {iterations}: {len(master_nodes)} clusters remaining.")


         if iterations >= max_iterations:
             logger.warning(f"Coalescing stopped after reaching max iterations ({max_iterations}).")

         logger.debug(f"Coalescing finished. Final number of unique stations: {len(master_nodes)}")

         # --- Update self.line_stations with coalesced NodeSpecs ---
         final_line_stations: Dict[str, List[NodeSpec]] = {}
         self.current_nodes = master_nodes # Store the final unique nodes

         for line_id, original_station_list in self.line_stations.items():
             coalesced_station_list = []
             seen_master_ids_in_line = set()
             for station in original_station_list:
                 master_id = station_map.get(station.id)
                 if master_id:
                     # Only add if this master node hasn't been added to *this line* yet
                     if master_id not in seen_master_ids_in_line:
                         master_node_spec = self.current_nodes[master_id]
                         coalesced_station_list.append(master_node_spec)
                         seen_master_ids_in_line.add(master_id)
                 else:
                     logger.warning(f"Original station {station.id} not found in final station_map during line update.")
             final_line_stations[line_id] = coalesced_station_list

         self.line_stations = final_line_stations


    def gen_edges(self):
        """Generates EdgeSpecs based on the connections within each line after coalescing."""
        self.current_edges = []
        added_edges = set() # Keep track of edges (node1_id, node2_id, line_id) to avoid duplicates

        for line_id, stations_on_line in self.line_stations.items():
            line_spec = next((line for line in self.line_set if line.id == line_id), None)
            if not line_spec:
                logger.warning(f"LineSpec not found for line ID {line_id} when generating edges.")
                continue

            for i in range(len(stations_on_line) - 1):
                station1 = stations_on_line[i]
                station2 = stations_on_line[i+1]

                # Skip if edge connects a station to itself (can happen after coalesce)
                if station1.id == station2.id:
                    continue

                # Ensure consistent order for edge uniqueness check
                node_ids = tuple(sorted((station1.id, station2.id)))
                edge_key = (node_ids[0], node_ids[1], line_id)

                if edge_key not in added_edges:
                    edge_spec = EdgeSpec(
                        station1=station1.id,
                        station2=station2.id,
                        line_id=line_spec.id,
                        line_name=line_spec.name,
                        line_color=line_spec.color,
                        line_stroke=line_spec.stroke,
                        # Add original line props if needed by functional ops
                        properties = {'line_has_aircon': line_spec.has_aircon, 'line_built': line_spec.built}
                    )
                    self.current_edges.append(edge_spec)
                    added_edges.add(edge_key)

        logger.debug(f"Generated {len(self.current_edges)} unique edges.")


    def use_int_names(self):
         """Renames stations and lines to use integer strings as names."""
         logger.debug("Renaming stations and lines to integers.")
         # --- Stations ---
         # Sort current nodes by some criteria (e.g., x, then y) for consistent naming if regenerated?
         sorted_nodes = sorted(self.current_nodes.values(), key=lambda s: (s.x, s.y))
         # Allocate more names than stations for fake name generation later
         station_int_names = list(range(len(sorted_nodes) * 2))
         random.shuffle(station_int_names)

         new_nodes = {}
         station_id_map = {} # Map old ID to new integer name (which becomes the new ID)
         for i, node in enumerate(sorted_nodes):
             old_id = node.id
             new_name = str(station_int_names[i])
             node.name = new_name
             node.id = new_name # IMPORTANT: Update ID to match name for consistency
             new_nodes[node.id] = node
             station_id_map[old_id] = new_name
         self.current_nodes = new_nodes

         # --- Lines ---
         sorted_lines = sorted(self.line_set, key=lambda l: l.name)
         line_int_names = list(range(len(sorted_lines) * 2)) # Use a different range?
         random.shuffle(line_int_names)
         line_id_map = {}
         for i, line in enumerate(sorted_lines):
              old_id = line.id
              new_name = str(line_int_names[i] + len(station_int_names)) # Offset line names
              line.name = new_name
              line.id = new_name # IMPORTANT: Update ID
              line_id_map[old_id] = new_name

         # --- Update Edges and Line Stations ---
         # Update station IDs in edges
         for edge in self.current_edges:
              edge.station1 = station_id_map.get(edge.station1, edge.station1) # Keep old if somehow missing map
              edge.station2 = station_id_map.get(edge.station2, edge.station2)
              # Update line ID and derived properties in edges
              new_line_id = line_id_map.get(edge.line_id, edge.line_id)
              edge.line_id = new_line_id
              # Update name/color/stroke from the potentially renamed line object
              renamed_line = next((l for l in self.line_set if l.id == new_line_id), None)
              if renamed_line:
                  edge.line_name = renamed_line.name
                  edge.line_color = renamed_line.color
                  edge.line_stroke = renamed_line.stroke
              else:
                   logger.warning(f"Could not find renamed line for ID {new_line_id} when updating edge.")


         # Update node objects in self.line_stations (dictionary values are lists of nodes)
         new_line_stations = {}
         for old_line_id, station_list in self.line_stations.items():
             new_line_id = line_id_map.get(old_line_id, old_line_id)
             new_station_list = []
             for station in station_list:
                 # Find the updated station object by its *new* ID (which is the int name)
                 new_station_id = station_id_map.get(station.id, station.id)
                 updated_station = self.current_nodes.get(new_station_id)
                 if updated_station:
                     new_station_list.append(updated_station)
                 else:
                      logger.warning(f"Could not find updated station for ID {new_station_id} in current_nodes.")
             new_line_stations[new_line_id] = new_station_list
         self.line_stations = new_line_stations


    def build_graph_spec(self):
        """Constructs the final GraphSpec object."""
        # Create line dictionary map: id -> LineSpec
        lines_dict = {line.id: line for line in self.line_set}

        # Nodes dictionary is already self.current_nodes (id -> NodeSpec)
        # Edges list is already self.current_edges

        if not self.current_nodes and self.current_edges:
            logger.warning("Graph generation resulted in edges but no nodes.")
            # Handle this case: maybe clear edges or raise error?
            self.current_edges = []
            self.current_nodes = {} # Ensure consistency


        self.graph_spec = GraphSpec(
            nodes=self.current_nodes,
            edges=self.current_edges,
            lines=lines_dict
        )
        logger.debug(f"Built GraphSpec with {len(self.graph_spec.nodes)} nodes and {len(self.graph_spec.edges)} edges.")


    def assert_data_valid(self):
        """Performs basic checks on the generated graph data."""
        if not self.graph_spec:
            raise ValueError("GraphSpec has not been built yet.")

        if not self.graph_spec.nodes and self.graph_spec.edges:
             logger.warning("Validation: Graph has edges but no nodes.")
             # Decide if this is critical - perhaps raise ValueError?

        node_ids = set(self.graph_spec.nodes.keys())
        line_ids = set(self.graph_spec.lines.keys())

        for edge in self.graph_spec.edges:
            if edge.station1 not in node_ids:
                raise ValueError(f"Edge references non-existent node ID: {edge.station1}")
            if edge.station2 not in node_ids:
                raise ValueError(f"Edge references non-existent node ID: {edge.station2}")
            if edge.line_id not in line_ids:
                 raise ValueError(f"Edge references non-existent line ID: {edge.line_id}")
            # Check derived edge props match line props
            line = self.graph_spec.lines[edge.line_id]
            if edge.line_name != line.name or edge.line_color != line.color or edge.line_stroke != line.stroke:
                 logger.warning(f"Edge properties mismatch LineSpec for line {edge.line_id}")

        if self.args.int_names:
            for node in self.graph_spec.nodes.values():
                try:
                    int(node.name)
                    if node.id != node.name:
                         raise ValueError(f"Integer name mismatch: node.id={node.id} != node.name={node.name}")
                except ValueError:
                    raise ValueError(f"Node name '{node.name}' is not an integer despite int_names=True.")
            for line in self.graph_spec.lines.values():
                 try:
                     int(line.name)
                     if line.id != line.name:
                         raise ValueError(f"Integer name mismatch: line.id={line.id} != line.name={line.name}")
                 except ValueError:
                     raise ValueError(f"Line name '{line.name}' is not an integer despite int_names=True.")

        logger.debug("Graph data validation passed.")
    
    def _get_line_info_for_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Helper to find line properties associated with a node's existing edges."""
        if not self.graph_spec: return None
        # Prioritize original edges before potentially added connector edges
        for edge in self.graph_spec.edges:
            if node_id == edge.station1 or node_id == edge.station2:
                line = self.graph_spec.lines.get(edge.line_id)
                if line:
                    return {
                        "line_id": line.id,
                        "line_name": line.name,
                        "line_color": line.color,
                        "line_stroke": line.stroke,
                        "properties": {'line_has_aircon': line.has_aircon, 'line_built': line.built}
                    }
        # Fallback if node has no edges (shouldn't happen in connected components > 1)
        logger.warning(f"Could not find existing line info for node {node_id}.")
        return None

    def ensure_connectivity(self):
        """Checks graph connectivity and adds edges between components if needed."""
        if not self.graph_spec:
            logger.error("Cannot ensure connectivity: GraphSpec not built.")
            return

        # Accessing .gnx builds the NetworkX graph if needed
        if nx.is_connected(self.graph_spec.gnx):
            logger.debug("Graph is already connected.")
            return

        logger.info("Graph is not connected. Adding edges to connect components...")
        components = list(nx.connected_components(self.graph_spec.gnx))
        num_components = len(components)
        logger.info(f"Found {num_components} connected components.")

        edges_added_count = 0
        # Connect components sequentially: comp 0 -> comp 1, comp 1 -> comp 2, ...
        for i in range(num_components - 1):
            comp1_nodes = list(components[i])
            comp2_nodes = list(components[i+1])

            # --- Pick nodes to connect (Randomly for simplicity) ---
            node1_id = random.choice(comp1_nodes)
            node2_id = random.choice(comp2_nodes)

            # --- Determine line properties for the new edge ---
            # Try getting from node1, then node2, then fallback to random line
            line_info = self._get_line_info_for_node(node1_id)
            if not line_info:
                line_info = self._get_line_info_for_node(node2_id)

            if not line_info:
                 # Highly unlikely fallback: pick a random existing line
                 if self.graph_spec.lines:
                     random_line = random.choice(list(self.graph_spec.lines.values()))
                     logger.warning(f"Using random line {random_line.id} for connector edge between {node1_id} and {node2_id}")
                     line_info = {
                         "line_id": random_line.id,
                         "line_name": random_line.name,
                         "line_color": random_line.color,
                         "line_stroke": random_line.stroke,
                         "properties": {'line_has_aircon': random_line.has_aircon, 'line_built': random_line.built}
                     }
                 else:
                     logger.error("Cannot add connector edge: No line info found and no lines exist in graph.")
                     continue # Skip adding this edge if no line info available

            # --- Create and add the new edge ---
            connector_edge = EdgeSpec(
                station1=node1_id,
                station2=node2_id,
                line_id=line_info["line_id"],
                line_name=line_info["line_name"] + " Connector" if not line_info["line_name"].endswith("Connector") else line_info["line_name"], # Add suffix
                line_color=line_info["line_color"],
                line_stroke="dotted", # Make connector edges visually distinct?
                properties=line_info["properties"]
            )

            # Check if this specific edge already exists (unlikely but possible)
            edge_key = tuple(sorted((node1_id, node2_id))) + (connector_edge.line_id,)
            if any(tuple(sorted((e.station1, e.station2))) + (e.line_id,) == edge_key for e in self.graph_spec.edges):
                logger.debug(f"Skipping duplicate connector edge: {edge_key}")
                continue


            self.graph_spec.edges.append(connector_edge)
            edges_added_count += 1
            logger.debug(f"Added connector edge between {node1_id} (comp {i}) and {node2_id} (comp {i+1}) on line {connector_edge.line_id}")

            # --- IMPORTANT: Invalidate the cached NetworkX graph ---
            # So it gets rebuilt with the new edge next time .gnx is accessed
            self.graph_spec._gnx = None

        logger.info(f"Added {edges_added_count} edges to connect components.")

        # Final verification (optional but recommended)
        if not nx.is_connected(self.graph_spec.gnx):
            logger.error("Failed to connect all graph components after adding edges!")
        else:
            logger.debug("Graph connectivity verified after adding edges.")


    def generate(self) -> 'GraphGenerator':
        """Runs the full graph generation pipeline."""
        logger.debug("Starting graph generation...")
        if self.args.mixed:
            size_id = random.choice([0, 1, 2])
            if size_id == 0:
                print("Generate small")
            elif size_id == 1:
                print("Generate medium")
            else:
                print("Generate large")
            self.gen_lines(size_id)
            self.gen_stations_along_lines(size_id)
            self.coalesce_nearby_stations(size_id)
            self.gen_edges()
        else:
            self.gen_lines()
            self.gen_stations_along_lines()
            self.coalesce_nearby_stations()
            self.gen_edges()

        if self.args.int_names:
            self.use_int_names()

        self.build_graph_spec()
        self.assert_data_valid()

        # --- Add connectivity step ---
        if not self.args.disconnected:
            self.ensure_connectivity()
        # --- End of added step ---

        logger.debug("Graph generation complete.")
        return self # Return self for potential chaining

    def draw(self, filename="./graph.png"):
        """Draws the generated graph using matplotlib."""
        if not self.graph_spec:
            logger.error("Cannot draw graph, GraphSpec not generated yet.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg") # Non-interactive backend
            from matplotlib import pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not found, cannot draw graph.")
            return

        logger.debug(f"Drawing graph to {filename}")
        fig, ax = plt.subplots(figsize=(20, 20)) # Reduced size slightly

        # Use the final coalesced nodes and edges
        nodes = self.graph_spec.nodes
        edges = self.graph_spec.edges
        lines = self.graph_spec.lines

        if not nodes:
             logger.warning("Graph has no nodes to draw.")
             plt.close(fig)
             return

        # Plot edges first, grouped by line
        lines_drawn = set()
        for line_id, line_spec in lines.items():
            line_edges = [edge for edge in edges if edge.line_id == line_id]
            if not line_edges: continue

            color = 'tab:'+line_spec.color if line_spec.color in plt.colormaps.get('tab10').colors else 'grey' # Fallback color
            ls_map = {'solid': '-', 'dashed': '--', 'dashdot': '-.', 'dotted': ':'}
            linestyle = ls_map.get(line_spec.stroke, '-') # Fallback linestyle

            # Plot segments for this line
            for edge in line_edges:
                node1 = nodes.get(edge.station1)
                node2 = nodes.get(edge.station2)
                if node1 and node2:
                    ax.plot([node1.x, node2.x], [node1.y, node2.y],
                            color=color, linestyle=linestyle, lw=3, marker='', zorder=1) # Edges below nodes
            lines_drawn.add(line_id)


        # Plot nodes
        node_positions = {nid: (node.x, node.y) for nid, node in nodes.items()}
        node_colors = []
        node_sizes = []
        is_interchange = Counter()

        # Determine interchanges (connected to >1 line or >2 edges)
        degree = Counter()
        lines_per_node = defaultdict(set)
        for edge in edges:
            degree[edge.station1] += 1
            degree[edge.station2] += 1
            lines_per_node[edge.station1] |= {edge.line_id} # Use set union
            lines_per_node[edge.station2] |= {edge.line_id}

        node_id_list = list(nodes.keys())
        for nid in node_id_list:
            is_inter = len(lines_per_node[nid]) > 1
            # Simple size/color logic (can be customized)
            node_colors.append('red' if is_inter else 'blue')
            node_sizes.append(80 if is_inter else 40)

        # Extract positions in the order of node_id_list
        pos_array = np.array([node_positions[nid] for nid in node_id_list])

        # Draw nodes using scatter for efficiency
        ax.scatter(pos_array[:, 0], pos_array[:, 1], s=node_sizes, c=node_colors, zorder=2)

        # Add labels (can be slow for large graphs)
        if len(nodes) < 100: # Only label smaller graphs
             for nid, pos in node_positions.items():
                 ax.text(pos[0], pos[1] + 0.1, nodes[nid].name, fontsize=8, ha='center')


        ax.set_title("Generated Transit Graph")
        ax.set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])

        try:
             plt.savefig(filename, bbox_inches='tight', dpi=150)
             logger.debug(f"Graph saved to {filename}")
        except Exception as e:
             logger.error(f"Failed to save graph image: {e}")
        finally:
             plt.close(fig) # Release memory


# --- Main Execution Guard ---
if __name__ == "__main__":
    args = get_args() # Assumes args.py is present and provides get_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # Set level specifically for this module's logger if needed
    # logger.setLevel(log_level)

    logger.info("Running GraphGenerator standalone test...")
    
    c = 1000
    n = 0
    tot_nodes = 0
    tot_edges = 0
    tot_lines = 0
    for i in range(c):
        generator = GraphGenerator(args)
        try:
            generator.generate()
            logger.info("Graph generation successful.")
            tot_nodes += len(generator.graph_spec.nodes)
            tot_edges += len(generator.graph_spec.edges)
            tot_lines += len(generator.graph_spec.lines)
            n += 1
            if args.draw:
                generator.draw("./generated_graph_test.png")
            # Print some stats
            
            if generator.graph_spec:
                print(f"Generated graph spec:")
                print(f"  Nodes: {len(generator.graph_spec.nodes)}")
                print(f"  Edges: {len(generator.graph_spec.edges)}")
                print(f"  Lines: {len(generator.graph_spec.lines)}")
        except Exception as e:
            logger.error(f"Graph generation failed: {e}", exc_info=True)
    
    print("Number of graphs:", n)
    print("Average nodes:", tot_nodes/n)
    print("Average edges:", tot_edges/n)
    print("Average lines:", tot_lines/n)