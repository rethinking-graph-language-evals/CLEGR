from transformers import AutoTokenizer
from src.data.evaluators.evaluator import DatasetEvaluator

class ArxivEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.class_names = [
            'arxiv cs.na', 'arxiv cs.mm', 'arxiv cs.lo', 'arxiv cs.cy', 'arxiv cs.cr',
            'arxiv cs.dc', 'arxiv cs.hc', 'arxiv cs.ce', 'arxiv cs.ni', 'arxiv cs.cc',
            'arxiv cs.ai', 'arxiv cs.ma', 'arxiv cs.gl', 'arxiv cs.ne', 'arxiv cs.sc',
            'arxiv cs.ar', 'arxiv cs.cv', 'arxiv cs.gr', 'arxiv cs.et', 'arxiv cs.sy',
            'arxiv cs.cg', 'arxiv cs.oh', 'arxiv cs.pl', 'arxiv cs.se', 'arxiv cs.lg',
            'arxiv cs.sd', 'arxiv cs.si', 'arxiv cs.ro', 'arxiv cs.it', 'arxiv cs.pf',
            'arxiv cs.cl', 'arxiv cs.ir', 'arxiv cs.ms', 'arxiv cs.fl', 'arxiv cs.ds',
            'arxiv cs.os', 'arxiv cs.gt', 'arxiv cs.db', 'arxiv cs.dl', 'arxiv cs.dm'
        ]

        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}
        self.class_to_label["none"] = len(self.class_names)


    def match_prediction(self, prediction: str) -> str:
        # get the part after the last dot, uppercase for uniformity
        suffix = prediction.split('.')[-1].strip().upper()

        for gt in self.class_names:
            if gt.split('.')[-1].strip().upper() == suffix:
                return gt
        return "none"
    
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
        
        matched = self.match_prediction(llm_output)
        p_lbl = self.class_to_label[matched]
        g_lbl = self.class_to_label[gt_str.lower()]

        # register for later metric computation
        self.add(p_lbl, g_lbl)
        return p_lbl
