import csv
import io
from typing import Dict, List, Any, Optional
import uuid
import networkx as nx
import torch
from torch_geometric.data import Data
from dataclasses import dataclass, field

# Define simple dataclasses for specs for the Interconnected System Grid
@dataclass
class NodeSpec:
    id: str
    name: str
    status: str
    security_level: str
    location_sector: str
    firmware_version: str
    power_consumption_units: int
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
    properties: Dict[str, Any] = field(default_factory=dict) # For all edge attributes

    def __getitem__(self, key):
        # All custom properties are in the properties dict for this schema
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.properties:
             return self.properties[key]
        else:
             raise KeyError(f"Key '{key}' not found in EdgeSpec")

# LineSpec is no longer used in the System Grid domain.

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
    def __init__(self, nodes: Dict[str, NodeSpec], edges: List[EdgeSpec], lines: Optional[Dict[str, Any]] = None):
        self.id: str = str(uuid.uuid4())
        self.nodes: Dict[str, NodeSpec] = nodes # Map node_id -> NodeSpec
        self.edges: List[EdgeSpec] = edges
        self.lines: Dict[str, Any] = lines or {} # No lines in this domain, but keep for compatibility
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
        }
        # Collect all unique values for each categorical feature
        node_cats = {"status": set(), "security_level": set(), "location_sector": set(), "firmware_version": set()}
        edge_cats = {"encryption_status": set()}

        for node in self.nodes.values():
            for key in node_cats:
                node_cats[key].add(node[key])

        for edge in self.edges:
             for key in edge_cats:
                 if key in edge.properties:
                    edge_cats[key].add(edge.properties[key])

        # Create integer mappings
        for key, values in node_cats.items():
            mappers["node"][key] = {val: i for i, val in enumerate(sorted(list(values)))}
        for key, values in edge_cats.items():
            mappers["edge"][key] = {val: i for i, val in enumerate(sorted(list(values)))}

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
            # Numerical features first
            feats = [
                float(node.power_consumption_units),
            ]
            # Then categorical features
            for key in ['status', 'security_level', 'location_sector', 'firmware_version']:
                feats.append(float(feature_mappers['node'][key][node[key]] + 1))
            node_features.append(feats)
        x = torch.tensor(node_features, dtype=torch.float)

        # --- Edge Index & Features ---
        src, dst, edge_attrs = [], [], []
        for edge in self.edges:
            u, v = node_id_to_idx.get(edge.station1), node_id_to_idx.get(edge.station2)
            if u is None or v is None: continue
            src.extend([u, v])
            dst.extend([v, u])

            # Numerical features first
            feats = [
                float(edge.properties.get('bandwidth_units', 0)),
                float(edge.properties.get('latency_ms', 0)),
            ]
            # Then categorical features
            for key in ['encryption_status']:
                if key in edge.properties:
                    feats.append(float(feature_mappers['edge'][key][edge.properties[key]] + 1))
                else:
                    feats.append(0.0) # Default value if key is missing
            
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
        node_writer.writerow(["id", "name", "status", "security_level", "location_sector", "firmware_version", "power_consumption_units"])
        for node in node_list:
            node_writer.writerow([
                node_id_to_idx[node.id],
                getattr(node, "name", None),
                node.status,
                node.security_level,
                node.location_sector,
                node.firmware_version,
                node.power_consumption_units
            ])

        # Write edges CSV
        edge_writer = csv.writer(edges_csv)
        edge_writer.writerow(["source_id", "target_id", "bandwidth_units", "latency_ms", "encryption_status"])
        for edge in self.edges:
            u, v = node_id_to_idx.get(edge.station1), node_id_to_idx.get(edge.station2)
            if u is None or v is None: continue
            edge_writer.writerow([
                u,
                v,
                edge.properties.get('bandwidth_units'),
                edge.properties.get('latency_ms'),
                edge.properties.get('encryption_status'),
            ])

        # Combine into one string
        combined_csv = (
            "--- System Nodes ---\n" + nodes_csv.getvalue() +
            "\n--- Links ---\n" + edges_csv.getvalue()
        )

        # Close buffers
        nodes_csv.close()
        edges_csv.close()

        # --- Build node_texts and edge_texts for BERT ---
        node_texts = []
        for node in node_list:
            name = getattr(node, "name", node.id)
            sentence = (
                f"System node {name} is in {node.location_sector} with status {node.status}. "
                f"It has security level {node.security_level}, firmware {node.firmware_version}, "
                f"and consumes {node.power_consumption_units} power units."
            )
            node_texts.append(sentence)

        edge_texts = []
        for edge in self.edges:
            src_name = getattr(self.nodes[edge.station1], "name", edge.station1)
            dst_name = getattr(self.nodes[edge.station2], "name", edge.station2)
            props = edge.properties
            sentence = (
                f"A link connects {src_name} and {dst_name}. "
                f"It has {props.get('bandwidth_units')} bandwidth units, {props.get('latency_ms')}ms latency, "
                f"and its encryption status is {props.get('encryption_status')}."
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