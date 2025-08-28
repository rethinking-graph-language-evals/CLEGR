import uuid
import random
import numpy as np
import bezier
from gibberish import Gibberish
from sklearn.neighbors import KDTree
import logging
from collections import Counter, defaultdict
import networkx as nx

from typing import List, Dict, Set, Optional, Any

logger = logging.getLogger(__name__)

from .types_computers import GraphSpec, NodeSpec, EdgeSpec
from .args import *

gib = Gibberish()

# --- Constants for Interconnected System Grid Generation ---
SystemNodeProperties = {
    "status": ['Operational', 'Degraded', 'Offline', 'Maintenance', 'Overloaded'],
    "security_level": ['Public', 'Internal', 'Confidential', 'Restricted'],
    "location_sector": ['Sector_Red', 'Sector_Blue', 'Sector_Green', 'Sector_Yellow', 'Sector_Purple'],
    "firmware_version": ['v1.0', 'v1.1', 'v2.0', 'v2.1', 'v3.0'],
    "power_consumption_units": [5, 15, 30, 75],
}

EdgeProperties = {
    "bandwidth_units": [10, 100, 1000, 5000],
    "latency_ms": [5, 20, 50, 100, 200],
    "encryption_status": ['Encrypted', 'Unencrypted', 'Partial'],
}

OtherProperties = {
    "name_prefix": ["Grid_Unit", "Core_Node", "Relay", "Nexus_Point", "Aether_Conduit", "Chrono_Nexus", "Quantum_Relay", "Echo_Chamber"],
    "name_suffix": ["Computer", "PC", "Machine", "comp", "device", "phone", "supercomputer"]
}

# --- Helper Functions ---
def gen_n(base, noise = 0.2):
    """Generates a number based on base value with Gaussian noise."""
    return max(1, round(random.gauss(base, noise*base)))

def add_noise(base, noise=0.05):
    """Adds uniform noise to a base value."""
    return base * (1 - noise + random.random() * noise*2)

# --- Graph Generator Class ---
class GraphGenerator:

    def __init__(self, args):
        self.args = args

        # Default statistics for the grid
        self.stats = {
            "nodes": 180,
            "avg_degree": 3,
            "map_radius": 25,
            "min_station_dist": 0.75, # Renamed from min_station_dist for consistency
            "coalesce_iterations": 15,
        }

        # Adjust stats based on args
        if args.medium:
            logger.info("Using medium graph settings.")
            self.stats["nodes"] = 90
            self.stats["avg_degree"] = 3
            self.stats["map_radius"] = 15
            self.stats["min_station_dist"] = 0.8
            self.stats["coalesce_iterations"] = 12
        elif args.small:
            logger.info("Using small graph settings.")
            self.stats["nodes"] = 25
            self.stats["avg_degree"] = 2
            self.stats["map_radius"] = 8
            self.stats["min_station_dist"] = 1.0
            self.stats["coalesce_iterations"] = 10
        elif args.mixed:
            logger.info("Using mixed graph settings.")
            self.stats["nodes"] = [180, 90, 25]
            self.stats["avg_degree"] = [3, 3, 2]
            self.stats["map_radius"] = [25, 15, 8]
            self.stats["min_station_dist"] = [0.75, 0.8, 1]
            self.stats["coalesce_iterations"] = [15, 12, 10]

        # Internal state during generation
        self.node_gen_set: Set[NodeSpec] = set()
        self.current_nodes: Dict[str, NodeSpec] = {}
        self.current_edges: List[EdgeSpec] = []
        self.graph_spec: Optional[GraphSpec] = None

    def _generate_id(self):
         """Generates a unique ID."""
         return str(uuid.uuid4())

    def _generate_name(self, prefix: str, suffix_pool: List[str]) -> str:
        """Generates a plausible name."""
        name = prefix + random.choice(suffix_pool)
        return name.title() # Capitalize words

    def gen_node(self) -> NodeSpec:
        """Generates a single SystemNode with random properties."""
        node_id = self._generate_id()
        props = {key: random.choice(values) for key, values in SystemNodeProperties.items()}
        name = self._generate_name(gib.generate_word()+"_", OtherProperties["name_suffix"])
        return NodeSpec(id=node_id, name=name, x=0.0, y=0.0, **props)

    def _get_unique_node(self) -> NodeSpec:
        """Generates a node, ensuring its name is unique within this generation."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            node = self.gen_node()
            is_unique = all(existing.name != node.name for existing in self.node_gen_set)
            if is_unique:
                 self.node_gen_set.add(node)
                 return node
            attempts += 1
        raise RuntimeError(f"Failed to generate a unique node name after {max_attempts} attempts.")

    def gen_nodes(self):
        """Generates the set of nodes for the graph and positions them."""
        self.node_gen_set = set()
        target_nodes = gen_n(self.stats["nodes"])
        map_radius = self.stats["map_radius"]

        for _ in range(target_nodes):
            node = self._get_unique_node()
            node.x = (random.random() * 2 - 1) * map_radius
            node.y = (random.random() * 2 - 1) * map_radius
        logger.debug(f"Generated {len(self.node_gen_set)} unique nodes.")

    def coalesce_nearby_nodes(self):
         """Merges nodes that are physically close to each other to form hubs."""
         min_dist = self.stats["min_station_dist"]
         if min_dist <= 0:
              logger.debug("Skipping node coalescing (min_dist <= 0).")
              self.current_nodes = {n.id: n for n in self.node_gen_set}
              return

         logger.debug(f"Starting node coalescing (min_dist={min_dist})...")
         all_nodes_list = list(self.node_gen_set)
         if len(all_nodes_list) < 2:
             self.current_nodes = {n.id: n for n in all_nodes_list}
             return

         node_map: Dict[str, str] = {s.id: s.id for s in all_nodes_list}
         master_nodes: Dict[str, NodeSpec] = {s.id: s for s in all_nodes_list}

         iterations = 0
         max_iterations = self.stats["coalesce_iterations"]
         while iterations < max_iterations:
             iterations += 1
             current_master_ids = list(master_nodes.keys())
             if len(current_master_ids) < 2: break

             points = np.array([[master_nodes[mid].x, master_nodes[mid].y] for mid in current_master_ids])
             master_id_map = {i: mid for i, mid in enumerate(current_master_ids)}

             tree = KDTree(points, leaf_size=10)
             nearby_pairs_indices = tree.query_radius(points, r=min_dist)

             parent = {mid: mid for mid in current_master_ids}
             def find_set(v):
                 if v == parent[v]: return v
                 parent[v] = find_set(parent[v])
                 return parent[v]

             def unite_sets(a, b):
                 a_root, b_root = find_set(a), find_set(b)
                 if a_root != b_root:
                     parent[b_root] = a_root
                     return True
                 return False

             merges_done = 0
             for i, neighbors in enumerate(nearby_pairs_indices):
                 for j in neighbors:
                     if i < j:
                         if unite_sets(master_id_map[i], master_id_map[j]):
                             merges_done += 1
             
             if merges_done == 0:
                 logger.debug("No further merges in this iteration.")
                 break

             new_master_nodes = {}
             for original_node_id in node_map:
                 root_master = find_set(node_map[original_node_id])
                 node_map[original_node_id] = root_master
                 if root_master not in new_master_nodes:
                     new_master_nodes[root_master] = master_nodes[root_master]
             master_nodes = new_master_nodes
             logger.debug(f"Iteration {iterations}: {len(master_nodes)} clusters remaining.")

         self.current_nodes = master_nodes
         logger.debug(f"Coalescing finished. Final number of unique nodes: {len(self.current_nodes)}")

    def gen_edges(self):
        """Generates edges between nearby nodes after coalescing."""
        self.current_edges = []
        added_edges = set()

        if len(self.current_nodes) < 2:
            logger.debug("Not enough nodes to generate edges.")
            return

        node_list = list(self.current_nodes.values())
        points = np.array([[node.x, node.y] for node in node_list])
        node_id_map = {i: node.id for i, node in enumerate(node_list)}

        tree = KDTree(points)
        k = self.stats["avg_degree"] + 1
        distances, indices = tree.query(points, k=min(k, len(node_list)))

        for i, neighbor_indices in enumerate(indices):
            node1_id = node_id_map[i]
            for j in neighbor_indices:
                if i == j: continue
                node2_id = node_id_map[j]

                edge_key = tuple(sorted((node1_id, node2_id)))
                if edge_key not in added_edges:
                    props = {key: random.choice(values) for key, values in EdgeProperties.items()}
                    edge_spec = EdgeSpec(station1=node1_id, station2=node2_id, properties=props)
                    self.current_edges.append(edge_spec)
                    added_edges.add(edge_key)
        logger.debug(f"Generated {len(self.current_edges)} unique edges.")

    def use_int_names(self):
         """Renames nodes to use integer strings as names."""
         logger.debug("Renaming nodes to integers.")
         sorted_nodes = sorted(self.current_nodes.values(), key=lambda s: (s.x, s.y))
         node_int_names = list(range(len(sorted_nodes) * 2))
         random.shuffle(node_int_names)

         new_nodes, node_id_map = {}, {}
         for i, node in enumerate(sorted_nodes):
             old_id, new_name = node.id, str(node_int_names[i])
             node.name = new_name
             node.id = new_name
             new_nodes[node.id] = node
             node_id_map[old_id] = new_name
         self.current_nodes = new_nodes

         for edge in self.current_edges:
              edge.station1 = node_id_map.get(edge.station1, edge.station1)
              edge.station2 = node_id_map.get(edge.station2, edge.station2)

    def build_graph_spec(self):
        """Constructs the final GraphSpec object."""
        self.graph_spec = GraphSpec(nodes=self.current_nodes, edges=self.current_edges, lines={})
        logger.debug(f"Built GraphSpec with {len(self.graph_spec.nodes)} nodes and {len(self.graph_spec.edges)} edges.")

    def assert_data_valid(self):
        """Performs basic checks on the generated graph data."""
        if not self.graph_spec: raise ValueError("GraphSpec not built.")
        node_ids = set(self.graph_spec.nodes.keys())
        for edge in self.graph_spec.edges:
            if edge.station1 not in node_ids: raise ValueError(f"Edge references non-existent node: {edge.station1}")
            if edge.station2 not in node_ids: raise ValueError(f"Edge references non-existent node: {edge.station2}")
        logger.debug("Graph data validation passed.")
    
    def _get_random_edge_properties(self) -> Dict[str, Any]:
        """Helper to generate a dictionary of random edge properties."""
        return {key: random.choice(values) for key, values in EdgeProperties.items()}

    def ensure_connectivity(self):
        """Checks graph connectivity and adds edges between components if needed."""
        if not self.graph_spec or self.args.disconnected: return
        if nx.is_connected(self.graph_spec.gnx):
            logger.debug("Graph is already connected.")
            return

        logger.info("Graph is not connected. Adding edges to connect components...")
        components = list(nx.connected_components(self.graph_spec.gnx))
        
        for i in range(len(components) - 1):
            node1_id = random.choice(list(components[i]))
            node2_id = random.choice(list(components[i+1]))

            connector_edge = EdgeSpec(station1=node1_id, station2=node2_id, properties=self._get_random_edge_properties())
            connector_edge.properties["is_connector"] = True
            
            edge_key = tuple(sorted((node1_id, node2_id)))
            if any(tuple(sorted((e.station1, e.station2))) == edge_key for e in self.graph_spec.edges):
                continue
            
            self.graph_spec.edges.append(connector_edge)
            self.graph_spec._gnx = None
        
        logger.info(f"Added {len(components) - 1} edges to connect components.")
        if not nx.is_connected(self.graph_spec.gnx):
            logger.error("Failed to connect all graph components!")

    def generate(self) -> 'GraphGenerator':
        """Runs the full graph generation pipeline."""
        logger.debug("Starting system grid generation...")
        self.gen_nodes()
        self.coalesce_nearby_nodes()
        self.gen_edges()

        if self.args.int_names:
            self.use_int_names()

        self.build_graph_spec()
        self.assert_data_valid()
        self.ensure_connectivity()

        logger.debug("Graph generation complete.")
        return self

    def draw(self, filename="./graph.png"):
        """Draws the generated graph using matplotlib."""
        if not self.graph_spec: return
        try:
            import matplotlib; matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not found, cannot draw graph.")
            return

        logger.debug(f"Drawing graph to {filename}")
        fig, ax = plt.subplots(figsize=(20, 20))
        nodes, edges = self.graph_spec.nodes, self.graph_spec.edges
        if not nodes: return

        for edge in edges:
            node1, node2 = nodes.get(edge.station1), nodes.get(edge.station2)
            if node1 and node2:
                color_map = {'Encrypted': 'green', 'Unencrypted': 'red', 'Partial': 'orange'}
                color = color_map.get(edge.properties.get('encryption_status'), 'grey')
                ax.plot([node1.x, node2.x], [node1.y, node2.y], color=color, linestyle='-', lw=1.5, zorder=1)

        node_positions = {nid: (node.x, node.y) for nid, node in nodes.items()}
        status_color_map = {'Operational': 'blue', 'Degraded': 'yellow', 'Offline': 'black', 'Maintenance': 'purple', 'Overloaded': 'red'}
        
        node_id_list = list(nodes.keys())
        node_colors = [status_color_map.get(nodes[nid].status, 'grey') for nid in node_id_list]
        node_sizes = [20 + nodes[nid].power_consumption_units * 2 for nid in node_id_list]
        pos_array = np.array([node_positions[nid] for nid in node_id_list])
        
        ax.scatter(pos_array[:, 0], pos_array[:, 1], s=node_sizes, c=node_colors, zorder=2)

        if len(nodes) < 100:
             for nid, pos in node_positions.items():
                 ax.text(pos[0], pos[1] + 0.1, nodes[nid].name, fontsize=8, ha='center')

        ax.set_title("Generated Interconnected System Grid")
        ax.set_aspect('equal', adjustable='box')
        plt.xticks([]); plt.yticks([])
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)

# --- Main Execution Guard ---
if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Running System Grid GraphGenerator standalone test...")
    generator = GraphGenerator(args)
    try:
        generator.generate()
        logger.info("Graph generation successful.")
        if args.draw:
            generator.draw("./generated_system_grid_test.png")
        if generator.graph_spec:
             print(f"Generated graph spec:\n  Nodes: {len(generator.graph_spec.nodes)}\n  Edges: {len(generator.graph_spec.edges)}")
    except Exception as e:
        logger.error(f"Graph generation failed: {e}", exc_info=True)