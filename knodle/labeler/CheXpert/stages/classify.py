"""Define mention classifier class."""
import logging
from negbio.pipeline import parse, ptb2ud, negdetect
from negbio.neg import semgraph, propagator, neg_detector
from negbio import ngrex
import networkx as nx
from .utils import *


class ModifiedDetector(neg_detector.Detector):
    """Child class of NegBio Detector class.

    Overrides parent methods __init__, detect, and match_uncertainty.
    """
    def __init__(self, config: Type[ChexpertConfig]):
        self.labeler_config = config
        self.neg_patterns = ngrex.load(self.labeler_config.neg_path)
        self.uncertain_patterns = ngrex.load(self.labeler_config.post_neg_unc_path)
        self.preneg_uncertain_patterns = ngrex.load(self.labeler_config.pre_neg_unc_path)

    def detect(self, sentence: Type[bioc.BioCSentence], locs: list) -> None:
        """Detect rules in report sentences.

        Args:
            sentence(BioCSentence): a sentence with universal dependencies
            locs(list): a list of (begin, end)

        Return:
            (str, MatcherObj, (begin, end)): negation or uncertainty,
            matcher, matched annotation
        """
        logger = logging.getLogger(__name__)

        try:
            g = semgraph.load(sentence)
            propagator.propagate(g)
        except Exception:
            logger.exception('Cannot parse dependency graph ' +
                             f'[offset={sentence.offset}]')
            raise
        else:
            for loc in locs:
                for node in neg_detector.find_nodes(g, loc[0], loc[1]):
                    # Pre-negation uncertainty rules are matched first.
                    preneg_m = self.match_prenegation_uncertainty(g, node)
                    if preneg_m:
                        yield self.labeler_config.uncertainty, preneg_m, loc
                    else:
                        # Then negation rules are matched.
                        neg_m = self.match_neg(g, node)
                        if neg_m:
                            yield self.labeler_config.negation, neg_m, loc
                        else:
                            # Finally, post-negation uncertainty rules are matched.
                            postneg_m = self.match_uncertainty(g, node)
                            if postneg_m:
                                yield self.labeler_config.uncertainty, postneg_m, loc

    def match_uncertainty(self,
                          graph: Type[nx.DiGraph],
                          node: Type[bioc.BioCNode]) -> Type[ngrex.pattern.MatcherObj]:
        for pattern in self.uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m

    def match_prenegation_uncertainty(self,
                                      graph: Type[nx.DiGraph],
                                      node: Type[bioc.BioCNode]) -> Type[ngrex.pattern.MatcherObj]:
        for pattern in self.preneg_uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m


class Classifier(object):
    """Classify mentions of observations from reports."""
    def __init__(self, config: Type[ChexpertConfig]):
        self.labeler_config = config
        self.parser = parse.NegBioParser(model_dir=self.labeler_config.parsing_model_dir)
        lemmatizer = ptb2ud.Lemmatizer()
        self.ptb2dep = ptb2ud.NegBioPtb2DepConverter(lemmatizer, universal=True)

        self.detector = ModifiedDetector(config=self.labeler_config)

    def classify(self, collection: Type[bioc.BioCCollection]) -> None:
        """Classify each mention into one of negative, uncertain, or none (positive)."""
        documents = collection.documents
        for document in documents:
            # Parse the impression text in place.
            self.parser.parse_doc(document)
            # Add the universal dependency graph in place.
            self.ptb2dep.convert_doc(document)
            # Detect the negation and uncertainty rules in place.
            negdetect.detect(document, self.detector)
            # To reduce memory consumption, remove sentences text.
            del document.passages[0].sentences[:]
