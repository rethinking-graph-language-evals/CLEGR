import re
from collections import defaultdict
import numpy as np

# Import necessary metrics from sklearn
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from .evaluator import DatasetEvaluator

class CLEGREvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.records = []

        # Central hub for question metadata (Scope & Multi-Sub-group)
        self.QUESTION_METADATA = {
            # ... [Full metadata dictionary from the previous step is assumed here] ...
            # --- Node Scope ---
            "StationPairAdjacent": {"scope": "Node", "subgroups": ["Aggregation", "Topology"]},
            "StationArchitectureAdjacent": {"scope": "Node", "subgroups": ["Filter", "Topology"]},
            "StationTwoHops": {"scope": "Node", "subgroups": ["Topology", "Aggregation"]},
            "HasCycle": {"scope": "Node", "subgroups": ["Topology"]},
            "StationOneApart": {"scope": "Node", "subgroups": ["Topology", "PathReasoning"]},
            "StationOneApartTrue": {"scope": "Node", "subgroups": ["Topology", "PathReasoning"]},
            "TopologyMostCommonArch": {"scope": "Node", "subgroups": ["Topology", "Aggregation"]},
            "CountIntersectionProperties": {"scope": "Node", "subgroups": ["Filter", "Aggregation"]},
            "CompareArchitectureCount": {"scope": "Node", "subgroups": ["Aggregation", "Filter"]},
            # --- Edge Scope ---
            "StationSameLineTrue": {"scope": "Edge", "subgroups": ["Topology", "Aggregation"]},
            "StationSameLine": {"scope": "Edge", "subgroups": ["Topology", "Aggregation"]},
            "EdgeFilterAirconCount": {"scope": "Edge", "subgroups": ["Filter", "Aggregation"]},
            "EdgeFilterColorCount": {"scope": "Edge", "subgroups": ["Filter", "Aggregation"]},
            "PathYearSpan": {"scope": "Edge", "subgroups": ["PathReasoning", "Aggregation"]},
            "PathOptimalColor": {"scope": "Edge", "subgroups": ["PathReasoning", "Aggregation"]},
            "PathEarliestBuilt": {"scope": "Edge", "subgroups": ["PathReasoning", "Aggregation"]},
            # --- Sub-graph Scope ---
            "LineTotalArchitectureCount": {"scope": "Sub-graph", "subgroups": ["Aggregation"]},
            "LineTotalMusicCount": {"scope": "Sub-graph", "subgroups": ["Aggregation"]},
            "LineTotalSizeCount": {"scope": "Sub-graph", "subgroups": ["Aggregation"]},
            "LineFilterMusicCount": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineFilterCleanlinessCount": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineFilterSizeCount": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineFilterDisabledAccessCount": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineFilterHasRailCount": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineStations": {"scope": "Sub-graph", "subgroups": ["Aggregation"]},
            "StationShortestCount": {"scope": "Sub-graph", "subgroups": ["PathReasoning", "Aggregation"]},
            "StationShortestAvoidingCount": {"scope": "Sub-graph", "subgroups": ["PathReasoning", "Filter"]},
            "StationShortestAvoidingArchitectureCount": {"scope": "Sub-graph", "subgroups": ["PathReasoning", "Filter"]},
            "DistinctRoutes": {"scope": "Sub-graph", "subgroups": ["Topology", "PathReasoning", "Aggregation"]},
            "CountEqualSizeStation": {"scope": "Sub-graph", "subgroups": ["Filter", "Aggregation"]},
            "LineIntersectionStations": {"scope": "Sub-graph", "subgroups": ["Topology", "Aggregation"]},
            "NodeOnPath": {"scope": "Sub-graph", "subgroups": ["PathReasoning"]},
            "PathMostCommonMusic": {"scope": "Sub-graph", "subgroups": ["PathReasoning", "Aggregation"]},
            "CompareLineDisabledAccess": {"scope": "Sub-graph", "subgroups": ["Aggregation", "Filter"]},
        }


        # Mapping question types to specific evaluation functions
        self.scorers = {
            #fact scorers below
            "StationPropertyCleanliness": self._exact_scorer,
            "StationPropertyCleanliness2": self._exact_scorer,
            "StationPropertySize": self._exact_scorer,
            "StationPropertySize2": self._exact_scorer,
            "StationPropertyMusic": self._exact_scorer,
            "StationPropertyMusic2": self._exact_scorer,
            "StationPropertyArchitecture": self._exact_scorer,
            "StationPropertyArchitecture2": self._exact_scorer,
            "StationPropertyDisabledAccess": self._boolean_scorer,
            "StationPropertyDisabledAccess2": self._boolean_scorer,
            "StationPropertyHasRail": self._boolean_scorer,
            "StationPropertyHasRail2": self._boolean_scorer,
            "StationExistence1": self._boolean_scorer,
            "StationExistence2": self._boolean_scorer,
            "StationLine": self._list_scorer,
            "StationLineCount": self._numeric_scorer,
            "StationAdjacent": self._boolean_scorer,
            "StationAdjacentAlwaysTrue": self._boolean_scorer,
            "EdgePropertyColor": self._exact_scorer,
            "EdgePropertyAircon": self._boolean_scorer,
            "EdgePropertyStroke": self._exact_scorer,
            "EdgePropertyBuilt": self._exact_scorer,
            #reasoning scorers below
            "StationPairAdjacent": self._exact_scorer, "StationArchitectureAdjacent": self._exact_scorer,
            "LineTotalArchitectureCount": self._numeric_scorer, "LineTotalMusicCount": self._numeric_scorer,
            "LineTotalSizeCount": self._numeric_scorer, "LineFilterMusicCount": self._numeric_scorer,
            "LineFilterCleanlinessCount": self._numeric_scorer, "LineFilterSizeCount": self._numeric_scorer,
            "LineFilterDisabledAccessCount": self._numeric_scorer, "LineFilterHasRailCount": self._numeric_scorer,
            "LineStations": self._list_scorer, "StationShortestCount": self._numeric_scorer,
            "StationShortestAvoidingCount": self._numeric_scorer, "StationShortestAvoidingArchitectureCount": self._numeric_scorer,
            "StationTwoHops": self._numeric_scorer, "DistinctRoutes": self._numeric_scorer,
            "HasCycle": self._boolean_scorer, "StationOneApartTrue": self._boolean_scorer,
            "StationOneApart": self._boolean_scorer, "StationSameLineTrue": self._boolean_scorer,
            "StationSameLine": self._boolean_scorer, "CountEqualSizeStation": self._numeric_scorer,
            "LineIntersectionStations": self._numeric_scorer, "NodeOnPath": self._boolean_scorer,
            "TopologyMostCommonArch": self._exact_scorer, "PathMostCommonMusic": self._exact_scorer,
            "CountIntersectionProperties": self._numeric_scorer, "EdgeFilterAirconCount": self._numeric_scorer,
            "EdgeFilterColorCount": self._numeric_scorer, "PathYearSpan": self._numeric_scorer,
            "PathOptimalColor": self._exact_scorer, "PathEarliestBuilt": self._numeric_scorer,
            "CompareLineDisabledAccess": self._exact_scorer, "CompareArchitectureCount": self._exact_scorer,
        }
        self._bool_true = {"yes", "true", "1"}
        self._bool_false = {"no", "false", "0"}

    # --- Normalization and Parsing functions (Unchanged) ---
    def normalize(self, s: str) -> str: return re.sub(r"[^\w\s]", "", s).lower().strip()
    def parse_list(self, s: str):
        if not s: return []
        items = re.split(r"[;,]\s*", str(s).strip())
        return [self.normalize(x) for x in items if x and self.normalize(x)]

    # --- Individual Scorers and Type Getters (Unchanged) ---
    def _get_scorer_type(self, qtype_str: str) -> str:
        scorer_func = self.scorers.get(qtype_str)
        if scorer_func == self._boolean_scorer: return "boolean"
        elif scorer_func == self._numeric_scorer: return "numeric"
        elif scorer_func == self._list_scorer: return "list"
        else: return "exact"
    def _exact_scorer(self, p, g): return self.normalize(p) == self.normalize(g)
    def _boolean_scorer(self, p, g):
        p_norm, g_norm = self.normalize(p), self.normalize(g)
        if g_norm in self._bool_true: return p_norm in self._bool_true
        if g_norm in self._bool_false: return p_norm in self._bool_false
        return False
    def _numeric_scorer(self, p, g):
        try: return np.isclose(float(p), float(g))
        except (ValueError, TypeError):
            p_clean = re.sub(r"[^\d\.\-eE]", "", str(p))
            g_clean = re.sub(r"[^\d\.\-eE]", "", str(g))
            try:
                if not p_clean or not g_clean: return False
                return np.isclose(float(p_clean), float(g_clean))
            except (ValueError, TypeError): return False
    def _list_scorer(self, p, g): return set(self.parse_list(p)) == set(self.parse_list(g))
    def _normalize_to_binary(self, s: str):
        norm = self.normalize(s)
        
        if norm in self._bool_true: return 1
        if norm in self._bool_false: return 0
        print(s, norm)
        return None
    def _normalize_to_float(self, s: str):
        try: return float(re.sub(r"[^\d\.\-eE]", "", str(s)))
        except (ValueError, TypeError): return None

    def __call__(self, data_obj, llm_output):
        pred_raw = llm_output.strip() if llm_output else ""
        print(data_obj)
        gt_raw = str(data_obj.label[0]).strip() if data_obj.label and data_obj.label[0] else ""
        qtype = data_obj.question_type[0]
        scorer = self.scorers.get(qtype, self._exact_scorer)
        correct = scorer(pred_raw, gt_raw)
        metadata = self.QUESTION_METADATA.get(qtype, {})
        self.records.append({
            "pred_raw": pred_raw, "gt_raw": gt_raw, "type": qtype,
            "group": data_obj.question_group[0], "correct": correct,
            "scope": metadata.get("scope", "Unknown"),
            "subgroups": metadata.get("subgroups", [data_obj.question_subgroup[0]])
        })
        return correct

    # --- Detailed Metric Calculation Helpers (Unchanged) ---
    def _calculate_boolean_metrics(self, gts_raw, preds_raw):
        gts_binary = [self._normalize_to_binary(g) for g in gts_raw]
        preds_binary = [self._normalize_to_binary(p) for p in preds_raw]
        valid_indices = [i for i, g in enumerate(gts_binary) if g is not None]
        if not valid_indices: return {"error": "No valid boolean ground truths"}
        gts_filt = [gts_binary[i] for i in valid_indices]
        preds_filt = [preds_binary[i] for i in valid_indices]
        preds_final = [p if p is not None else 1 - g for i, (p, g) in enumerate(zip(preds_filt, gts_filt))]
        if len(set(gts_filt)) <= 1:
            return {"accuracy": accuracy_score(gts_filt, preds_final)}
        return { "accuracy": accuracy_score(gts_filt, preds_final),
                 "balanced_accuracy": balanced_accuracy_score(gts_filt, preds_final),
                 "f1_score_macro": f1_score(gts_filt, preds_final, average="macro", zero_division=0),
                 "mcc": matthews_corrcoef(gts_filt, preds_final)}

    def _calculate_numeric_metrics(self, gts_raw, preds_raw):
        gts_numeric = [self._normalize_to_float(g) for g in gts_raw]
        preds_numeric = [self._normalize_to_float(p) for p in preds_raw]
        valid_pairs = [(g, p) for g, p in zip(gts_numeric, preds_numeric) if g is not None and p is not None]
        if not valid_pairs: return {"error": "No valid numeric pairs"}
        gts_filt, preds_filt = zip(*valid_pairs)
        exact_correct = sum(np.isclose(g, p) for g, p in valid_pairs)
        return { "exact_accuracy": exact_correct / len(valid_pairs),
                 "mae": mean_absolute_error(gts_filt, preds_filt),
                 "rmse": root_mean_squared_error(gts_filt, preds_filt),
                 "n_valid_numeric_pairs": len(valid_pairs) }

    def _calculate_list_metrics(self, gts_raw, preds_raw):
        f1s = []
        for p_raw, g_raw in zip(preds_raw, gts_raw):
            p_set, g_set = set(self.parse_list(p_raw)), set(self.parse_list(g_raw))
            if not g_set: f1 = 1.0 if not p_set else 0.0
            elif not p_set: f1 = 0.0
            else:
                common = len(p_set.intersection(g_set))
                prec = common / len(p_set)
                rec = common / len(g_set)
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return {"mean_set_f1": np.mean(f1s) if f1s else 0.0}

    def _calculate_exact_metrics(self, gts_raw, preds_raw):
        correct = sum(self._exact_scorer(p, g) for p, g in zip(preds_raw, gts_raw))
        total = len(gts_raw)
        return {"accuracy": correct / total if total else 0.0}

    # ======================================================================
    # NEW: Generic helper to generate rich, multi-metric reports for any category
    # ======================================================================
    def _get_rich_metrics_for_category(self, records):
        """Calculates detailed metrics for a list of records, bucketing by scorer type."""
        n_total = len(records)
        if n_total == 0: return {}

        # Bucket predictions and ground truths by scorer type
        preds_by_type = defaultdict(list)
        gts_by_type = defaultdict(list)
        for r in records:
            scorer_type = self._get_scorer_type(r["type"])
            preds_by_type[scorer_type].append(r["pred_raw"])
            gts_by_type[scorer_type].append(r["gt_raw"])

        # Calculate detailed metrics for each bucket
        type_metrics = {}
        for scorer_type, gts in gts_by_type.items():
            preds = preds_by_type[scorer_type]
            if scorer_type == "boolean": type_metrics["boolean"] = self._calculate_boolean_metrics(gts, preds)
            elif scorer_type == "numeric": type_metrics["numeric"] = self._calculate_numeric_metrics(gts, preds)
            elif scorer_type == "list": type_metrics["list"] = self._calculate_list_metrics(gts, preds)
            else: type_metrics["exact"] = self._calculate_exact_metrics(gts, preds)
            type_metrics[scorer_type]["n_examples"] = len(gts)

        return {
            "overall_accuracy": sum(int(r["correct"]) for r in records) / n_total,
            "n_examples": n_total,
            "type_metrics": type_metrics
        }

    def compute_metrics(self):
        """Computes a full suite of metrics, including detailed per-type and aggregated reports."""
        # --- 1. Bucket all records by their categories ---
        by_type = defaultdict(list)
        by_group = defaultdict(list)
        by_scope = defaultdict(list)
        by_subgroup = defaultdict(list)

        for r in self.records:
            by_type[r["type"]].append(r)
            by_group[r["group"]].append(r)
            by_scope[r["scope"]].append(r)
            for subgroup in r["subgroups"]:
                by_subgroup[subgroup].append(r)

        metrics = {"overall": {}, "by_type": {}, "by_group": {}, "by_scope": {}, "by_subgroup": {}}

        # --- 2. Calculate Overall Accuracy ---
        total_questions = len(self.records)
        if total_questions > 0:
            metrics["overall"]["accuracy"] = sum(int(r["correct"]) for r in self.records) / total_questions
            metrics["n_examples"] = total_questions

        # --- 3. Calculate Granular Per-Question-Type Metrics ---
        for qtype, records in by_type.items():
            scorer_type = self._get_scorer_type(qtype)
            gts = [r["gt_raw"] for r in records]
            preds = [r["pred_raw"] for r in records]
            if scorer_type == "boolean": res = self._calculate_boolean_metrics(gts, preds)
            elif scorer_type == "numeric": res = self._calculate_numeric_metrics(gts, preds)
            elif scorer_type == "list": res = self._calculate_list_metrics(gts, preds)
            else: res = self._calculate_exact_metrics(gts, preds)
            res["n_examples"] = len(records)
            metrics["by_type"][qtype] = res

        # --- 4. Calculate Rich Aggregated Metrics for Group, Scope, and Subgroup ---
        for group, records in by_group.items():
            metrics["by_group"][group] = self._get_rich_metrics_for_category(records)
        for scope, records in by_scope.items():
            metrics["by_scope"][scope] = self._get_rich_metrics_for_category(records)
        for subgroup, records in by_subgroup.items():
            metrics["by_subgroup"][subgroup] = self._get_rich_metrics_for_category(records)

        return metrics

