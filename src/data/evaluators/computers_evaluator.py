from transformers import AutoTokenizer
from src.data.evaluators.evaluator import DatasetEvaluator

class ComputersEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.class_names = [
            "computer accessories and peripherals",
            "tablet accessories",
            "laptop accessories",
            "computers and tablets",
            "computer components",
            "data storage",
            "networking products",
            "monitors",
            "servers",
            "tablet replacement parts"
        ]

        self.class_to_label = {c: i for i, c in enumerate(self.class_names)}
        self.class_to_label["None"] = len(self.class_names)

    def match_prediction(self, prediction: str, ground_truths: list[str]) -> str:
        p = prediction.strip()
        for truth in ground_truths:
            if p.startswith(truth):
                return truth
        return "None"
    
    def max_class_toklen(self, llm_tokenizer: AutoTokenizer) -> int:
        # Tokenize each class name separately
        tokenized_class_names = [llm_tokenizer(name, add_special_tokens=False).input_ids for name in self.class_names]

        if self._max_class_toklen is None:
            self._max_class_toklen = max(len(class_name_toks) for class_name_toks in tokenized_class_names)

            # Print all class names with their token length
            for class_name, class_name_toks in zip(self.class_names, tokenized_class_names):
                print(f"{class_name}: {len(class_name_toks)}")
                

        return self._max_class_toklen
    
    def __call__(self, data_obj: str, llm_output: str) -> int:
        """
        Process one example:
          pred_str: raw LLM output (string)
          data_obj:   the pyg data object
        Returns the predicted label index.
        """
        gt_str = data_obj.label
        if isinstance(gt_str, list):
            gt_str = gt_str[0]
            
        self.add_raw(llm_output, gt_str)
        
        matched = self.match_prediction(llm_output, self.class_names)
        p_lbl = self.class_to_label[matched]
        g_lbl = self.class_to_label[gt_str]

        # register for later metric computation
        self.add(p_lbl, g_lbl)
        return p_lbl
