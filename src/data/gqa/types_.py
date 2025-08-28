import csv
import io
from typing import Dict, List, Any, Optional
import uuid
import networkx as nx
import torch
from torch_geometric.data import Data
from dataclasses import dataclass, field

# Define simple dataclasses for specs
@dataclass
class NodeSpec:
    id: str
    name: str
    architecture: str
    cleanliness: str
    disabled_access: bool
    has_rail: bool
    music: str
    size: str
    x: float
    y: float
    # Allow extra fields if needed during generation
    properties: Dict[str, Any] = field(default_factory=dict)

    # Make hashable based on ID for use in sets/dicts
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, NodeSpec):
            return self.id == other.id
        return False

    # Helper to access properties easily
    def __getitem__(self, key):
         # Check standard fields first, then properties dict
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.properties:
             return self.properties[key]
        else:
             raise KeyError(f"Key '{key}' not found in NodeSpec")

@dataclass
class EdgeSpec:
    station1: str # Node ID
    station2: str # Node ID
    line_id: str
    line_name: str
    line_color: str
    line_stroke: str
    properties: Dict[str, Any] = field(default_factory=dict) # For any extra original properties

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.properties:
             return self.properties[key]
        else:
             raise KeyError(f"Key '{key}' not found in EdgeSpec")

@dataclass
class LineSpec:
    id: str
    name: str
    color: str
    stroke: str
    built: str
    has_aircon: bool
    properties: Dict[str, Any] = field(default_factory=dict) # For any extra original properties

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, LineSpec):
            return self.id == other.id
        return False

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.properties:
             return self.properties[key]
        else:
             raise KeyError(f"Key '{key}' not found in LineSpec")


@dataclass
class QuestionSpec:
    english: str
    functional: Dict[str, Any] # Keep the stripped functional representation
    type_id: int
    type_string: str
    group: Optional[str]
    subgroup: Optional[str]

    def __repr__(self):
        return self.english


class GraphSpec:
    """Holds the graph data and provides conversion to PyG format."""
    def __init__(self, nodes: Dict[str, NodeSpec], edges: List[EdgeSpec], lines: Dict[str, LineSpec]):
        self.id: str = str(uuid.uuid4())
        self.nodes: Dict[str, NodeSpec] = nodes # Map node_id -> NodeSpec
        self.edges: List[EdgeSpec] = edges
        self.lines: Dict[str, LineSpec] = lines # Map line_id -> LineSpec
        self._gnx: Optional[nx.Graph] = None # Lazy generation of networkx graph

    @property
    def gnx(self) -> nx.Graph:
        """Lazy generator for the NetworkX graph instance."""
        if self._gnx is None:
            self._gnx = nx.Graph()
            for node_id, node_spec in self.nodes.items():
                # Store the full NodeSpec object as attribute
                self._gnx.add_node(node_id, data=node_spec)
            for edge_spec in self.edges:
                 # Store the full EdgeSpec object as attribute
                self._gnx.add_edge(edge_spec.station1, edge_spec.station2, data=edge_spec)
        return self._gnx

    def get_feature_mappers(self) -> Dict[str, Dict[str, int]]:
        """Creates mappings from categorical features to integers."""
        mappers = {
            "node": {},
            "edge": {},
            "line": {}, # Although line props are on edges, useful to map globally
        }
        # Collect all unique values for each categorical feature
        node_cats = {"architecture": set(), "cleanliness": set(), "music": set(), "size": set()}
        edge_cats = {"line_color": set(), "line_stroke": set()} # Edge attributes directly from EdgeSpec
        line_cats = {"color": set(), "stroke": set(), "built": set()} # Attributes from LineSpec

        for node in self.nodes.values():
            for key in node_cats:
                node_cats[key].add(node[key])

        for edge in self.edges:
             for key in edge_cats:
                 edge_cats[key].add(edge[key])

        for line in self.lines.values():
             for key in line_cats:
                 if key in ["color", "stroke"]: # Handle potential overlap with edge_cats if needed
                     edge_cats[f"line_{key}"].add(line[key])
                 line_cats[key].add(line[key])


        # Create integer mappings
        for key, values in node_cats.items():
            mappers["node"][key] = {val: i for i, val in enumerate(sorted(list(values)))}
        for key, values in edge_cats.items():
             # Use prefix to avoid clashes if needed, e.g. edge_line_color
            mappers["edge"][key] = {val: i for i, val in enumerate(sorted(list(values)))}
        for key, values in line_cats.items():
            mappers["line"][key] = {val: i for i, val in enumerate(sorted(list(values)))}

        return mappers

    def to_pyg_data(self, feature_mappers: Dict[str, Dict[str, Dict[str, int]]]) -> Data:
        """
        Converts the GraphSpec to a torch_geometric.data.Data object, builds:
        - a JSON‐serialised `context` of the graph
        - `node_texts`: list of NL sentences describing each node
        - `edge_texts`: list of NL sentences describing each edge
        """
        if not self.nodes:
            data = Data()
            data.context = ""
            data.node_texts = []
            data.edge_texts = []
            return data

        node_list = list(self.nodes.values())
        node_id_to_idx = {node.id: i for i, node in enumerate(node_list)}

        # --- Node Features (x) ---
        node_features = []
        for node in node_list:
            feats = [
                float(node.disabled_access),
                float(node.has_rail),
            ]
            for key in ['architecture', 'cleanliness', 'music', 'size']:
                feats.append(float(feature_mappers['node'][key][node[key]] + 1))
            node_features.append(feats)
        x = torch.tensor(node_features, dtype=torch.float)

        # --- Edge Index & Features ---
        src, dst, edge_attrs = [], [], []
        for edge in self.edges:
            u, v = node_id_to_idx[edge.station1], node_id_to_idx[edge.station2]
            src.extend([u, v])
            dst.extend([v, u])

            feats = []
            for key in ['line_color', 'line_stroke']:
                feats.append(float(feature_mappers['edge'][key][edge[key]] + 1))
            line = self.lines.get(edge.line_id)
            if line:
                feats.append(float(line.has_aircon))
                feats.append(float(feature_mappers['line']['built'][line.built] + 1))
            else:
                feats.extend([0.0, 0.0])
            edge_attrs.append(feats)
            edge_attrs.append(feats)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = (torch.tensor(edge_attrs, dtype=torch.float)
                    if edge_attrs else torch.empty((0, 0), dtype=torch.float))

        # consistency checks
        if edge_attr.nelement() and edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(f"Shape mismatch: edge_attr {edge_attr.shape} vs edge_index {edge_index.shape}")
        if not edge_attr.nelement() and edge_index.shape[1] != 0:
            raise ValueError(f"Shape mismatch: edge_attr {edge_attr.shape} vs edge_index {edge_index.shape}")

        # Create CSV buffers
        nodes_csv = io.StringIO()
        edges_csv = io.StringIO()

        # Write nodes CSV
        node_writer = csv.writer(nodes_csv)
        node_writer.writerow(["id", "name", "disabled_access", "has_rail", "architecture", "cleanliness", "music", "size"])
        for node in node_list:
            node_writer.writerow([
                node_id_to_idx[node.id],
                getattr(node, "name", None),
                node.disabled_access,
                node.has_rail,
                node.architecture,
                node.cleanliness,
                node.music,
                node.size
            ])

        # Write edges CSV
        edge_writer = csv.writer(edges_csv)
        edge_writer.writerow(["source_id", "target_id", "line_color", "line_stroke", "has_aircon", "built"])
        for edge in self.edges:
            line = self.lines.get(edge.line_id)
            edge_writer.writerow([
                node_id_to_idx[edge.station1],
                node_id_to_idx[edge.station2],
                edge.line_color,
                edge.line_stroke,
                getattr(line, "has_aircon", None) if line else None,
                getattr(line, "built", None) if line else None
            ])

        # Combine into one string
        combined_csv = (
            "--- Nodes ---\n" + nodes_csv.getvalue() +
            "\n--- Edges ---\n" + edges_csv.getvalue()
        )

        # Close buffers
        nodes_csv.close()
        edges_csv.close()

        # --- Build node_texts and edge_texts for BERT ---
        node_texts = []
        for node in node_list:
            name = getattr(node, "name", node.id)
            da = "has disabled access" if node.disabled_access else "does not have disabled access"
            hr = "has rail"            if node.has_rail       else "does not have rail"
            arch = node.architecture
            clean = node.cleanliness
            music = node.music
            size = node.size
            sentence = (
                f"{name} {da} and {hr}. "
                f"It features {arch} architecture, has {clean} cleanliness, "
                f"{music} music, and is {size} in size."
            )
            node_texts.append(sentence)

        edge_texts = []
        for edge in self.edges:
            src_name = getattr(self.nodes[edge.station1], "name", edge.station1)
            dst_name = getattr(self.nodes[edge.station2], "name", edge.station2)
            lc = edge.line_color
            ls = edge.line_stroke
            line = self.lines.get(edge.line_id)
            if line:
                ac = "has air conditioning" if line.has_aircon else "does not have air conditioning"
                built = line.built
                sentence = (
                    f"There is a {ls} {lc} line from {src_name} to {dst_name}. "
                    f"It {ac} and was built in {built}."
                )
            else:
                sentence = (
                    f"There is a {ls} {lc} line from {src_name} to {dst_name}."
                )
            edge_texts.append(sentence)
            edge_texts.append(sentence)

        # --- Assemble Data object ---
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id    = self.id
        data.node_ids    = list(node_id_to_idx.keys())
        data.context     = combined_csv
        data.node_texts  = node_texts
        data.edge_texts  = edge_texts

        return data



# We don't need DocumentSpec anymore, as data is combined in generate.py
# class DocumentSpec(Strippable): ... (REMOVE)