
class Modifier:
    def __call__(self, question_inputs, context_inputs, graph_embs, tokenizer):
        raise NotImplementedError("Modifier must implement __call__ method")
    
    def __repr__(self):
        return self.__class__.__name__