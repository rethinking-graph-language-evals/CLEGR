import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate CLEGR Graph Question Answering Dataset in PyTorch Geometric Format")

    # --- Filtering Questions ---
    parser.add_argument('--group', type=str, default=None,
                        help="Only generate questions from a specific group (e.g., 'MultiStep')")
    parser.add_argument('--type-prefix', action="append",
                        help="Only generate questions whose type string starts with this prefix (can be specified multiple times)")

    # --- Output Control ---
    parser.add_argument('--name', type=str, default=None,
                        help="Custom name for the output dataset files (default: clegr_pyg_<uuid>)")
    parser.add_argument('--output-dir', type=str, default="./data_pyg",
                        help="Directory to save the generated PyG dataset and mappers")

    # --- Generation Size ---
    parser.add_argument('--count', type=int, default=500,
                        help="Target number of (Graph, Question, Answer) tuples to generate")
    parser.add_argument('--questions-per-graph', type=int, default=2,
                        help="Number of instances of each question signature")
    parser.add_argument('--just-one', action='store_true',
                        help="Generate only one GQA tuple (useful for debugging)")

    # --- Graph Generation Parameters ---
    parser.add_argument('--medium', action='store_true',
                        help="Generate medium graphs (12 lines, 12 stations)")
    parser.add_argument('--small', action='store_true',
                        help="Generate small graphs (5-6 lines, fewer stations)")
    parser.add_argument('--mixed', action='store_true',
                        help="Generate equal proportions of graphs of all sizes")
    parser.add_argument('--int-names', action='store_true', dest="int_names",
                        help="Use integers as station/line names and IDs instead of generated words/UUIDs")
    parser.add_argument('--disconnected', action='store_true', dest="disconnected",
                        help="Allow generating disconnected graphs (for testing purposes)")

    # --- Debugging & Logging ---
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument('--draw', action='store_true',
                        help="Draw images of the first few generated graphs (saved to output dir)")

    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Post-processing/Validation ---
    if args.medium and args.small:
        parser.error("Cannot specify both --medium and --small graph sizes.")

    return args