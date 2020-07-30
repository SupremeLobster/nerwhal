import importlib.util
import os
from collections import OrderedDict
from multiprocessing import Pipe
from multiprocessing.context import Process
from typing import List

import nerwhal.backends
from nerwhal.combination_strategies import combine
from nerwhal.backends.stanza_ner_backend import StanzaNerBackend
from nerwhal.tokenizer import Tokenizer
from nerwhal.scorer import score_entities
from nerwhal.types import Config, NamedEntity
from nerwhal.utils import add_token_indices

EXAMPLE_RECOGNIZERS_PATH = "nerwhal/example_recognizers"


class Core:
    def __init__(self):
        self.config = None
        self.backends = OrderedDict()
        self.tokenizer = None
        self.recognizer_lookup = None

    def update_config(self, config):
        """Whenever the config is changed, the state of core is rebuilt from scratch.

        :param config:
        :return:
        """
        if config == self.config:
            return

        self.config = config

        self.tokenizer = Tokenizer(self.config.language)

        self.backends = {}

        if self.config.use_statistical_ner:
            self.backends["stanza"] = StanzaNerBackend(self.config.language)

        if self.config.load_example_recognizers:
            self._add_examples_to_config_recognizer_paths()

        self.recognizer_lookup = {}
        for recognizer_path in self.config.recognizer_paths:
            if not os.path.isfile(recognizer_path):
                raise ValueError(f"Configured recognizer {recognizer_path} is not a file")

            recognizer_cls = self._load_class(recognizer_path)
            self.recognizer_lookup[recognizer_cls.__name__] = recognizer_cls

            # import only the backend modules that are configured
            if recognizer_cls.BACKEND not in self.backends.keys():
                backend_cls = nerwhal.backends.load(recognizer_cls.BACKEND)

                if recognizer_cls.BACKEND == "entity-ruler":
                    backend_inst = backend_cls(self.config.language)
                else:
                    backend_inst = backend_cls()

                self.backends[recognizer_cls.BACKEND] = backend_inst

            self.backends[recognizer_cls.BACKEND].register_recognizer(recognizer_cls)

    def _load_class(self, recognizer_path):
        module_name = os.path.splitext(os.path.basename(recognizer_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, recognizer_path)
        module = importlib.util.module_from_spec(spec)
        class_name = "".join(word.title() for word in module_name.split("_"))
        spec.loader.exec_module(module)
        recognizer_cls = getattr(module, class_name)
        return recognizer_cls

    def _add_examples_to_config_recognizer_paths(self):
        for file in os.listdir(EXAMPLE_RECOGNIZERS_PATH):
            if file.endswith("_recognizer.py"):
                example = os.path.join(EXAMPLE_RECOGNIZERS_PATH, file)
                if example not in self.config.recognizer_paths:
                    self.config.recognizer_paths.append(example)

    def run_recognition(self, text):
        list_of_ent_lists = self._run_in_parallel(self.backends.values(), text)
        return list_of_ent_lists

    def _run_in_parallel(self, backends, text):
        def target(func, arg, pipe_end):
            pipe_end.send(func(arg))

        jobs = []
        pipe_conns = []
        for backend in backends:
            recv_end, send_end = Pipe(False)
            proc = Process(target=target, args=(backend.run, text, send_end))
            jobs.append(proc)
            pipe_conns.append(recv_end)
            proc.start()
        for proc in jobs:
            proc.join()
        results = [conn.recv() for conn in pipe_conns]
        return results


core = Core()


def recognize(text: str, config: Config, combination_strategy="append", context_words=False, compute_tokens=True) -> dict:
    """Find personally identifiable data in the given text and return it.

    :param context_words: if True, use context words to boost the score of entities: if one of the
    :param compute_tokens:
    :param config:
    :param text:
    :param combination_strategy: choose from `append`, `disjunctive_union` and `fusion`
    """
    core.update_config(config)
    results = core.run_recognition(text)

    if len(results) == 0:
        ents = []
    else:
        ents = combine(*results, strategy=combination_strategy)

    result = {}
    tokens = []
    if compute_tokens or context_words:
        # tokenize
        core.tokenizer.tokenize(text)
        tokens = core.tokenizer.get_tokens()
        add_token_indices(ents, tokens)

    if compute_tokens:
        result["tokens"] = tokens

    if context_words:
        for ent in ents:
            sentence_tokens = core.tokenizer.get_sentence_for_token(ent.start_tok)
            sentence_without_ent = [
                token.text for token in sentence_tokens if token.i < ent.start_tok or token.i >= ent.end_tok
            ]
            context_words = core.recognizer_lookup[ent.recognizer].CONTEXT_WORDS
            if any(word in sentence_without_ent for word in context_words):
                ent.score = min(ent.score * core.config.context_word_confidence_boost_factor, 1.0)

    result["ents"] = ents
    return result


def evaluate(ents: List[NamedEntity], gold: List[NamedEntity]) -> dict:
    """Compute the scores of a list of recognized named entities compared to the corresponding true entities.

    Each named entity is required to have the fields `start_char`, `end_char` and `tag` populated. The remaining fields
    are ignored.
    """
    return score_entities(ents, gold)
