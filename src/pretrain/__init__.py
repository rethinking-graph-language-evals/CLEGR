from .tea_glm import TEAGLMTrainer
from .base_trainer import GNNTrainer

def get_trainer(method) -> GNNTrainer:
    mapper = {
        "tea-glm": TEAGLMTrainer
    }
    return mapper[method]