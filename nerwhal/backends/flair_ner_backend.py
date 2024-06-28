from nerwhal.backends.base import Backend
from nerwhal.types import NamedEntity

from flair.data import Sentence
from flair.models import SequenceTagger

class FlairNerBackend(Backend):
    """
    This backend recognizes entities using Flair's neural network models.
    """

    def __init__(self, model_path="pytorch_model.bin"):
        # load tagger
        self.flair_nlp = SequenceTagger.load(model_path)

    def register_recognizer(self, recognizer_cls):
        raise NotImplementedError()

    def run(self, text):
        sentence = Sentence(text)
        self.flair_nlp.predict(sentence)

        return [
            NamedEntity(ent.start_position, ent.end_position, ent.get_label().value,
                        ent.text, ent.get_label().score, self.__class__.__name__)
            for ent in sentence.get_spans("ner")
        ]
