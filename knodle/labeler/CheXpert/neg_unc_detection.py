"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------

Define negation and uncertainty detection class.
"""
import bioc
import logging

import networkx as nx

from negbio import ngrex
from negbio.neg import semgraph, propagator, neg_detector
from negbio.pipeline2 import parse, lemmatize, ptb2ud, negdetect

from .config import CheXpertConfig


class ModifiedDetector(neg_detector.Detector):
    """
    Child class of NegBio Detector class.
    Overrides parent methods __init__, detect, and match_uncertainty.

    See: https://github.com/stanfordmlgroup/chexpert-labeler/blob/master/stages/classify.py
    """
    def __init__(self, config: CheXpertConfig):
        self.labeler_config = config
        self.neg_patterns = ngrex.load_yml(self.labeler_config.neg_path)
        self.uncertain_patterns = ngrex.load_yml(self.labeler_config.post_neg_unc_path)
        self.preneg_uncertain_patterns = ngrex.load_yml(self.labeler_config.pre_neg_unc_path)

    def detect(self, sentence: bioc.BioCSentence, locs: list) -> None:
        """Detect rules in report sentences. Return negation or uncertainty if detectable."""
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
                          graph: nx.DiGraph,
                          node: bioc.BioCNode) -> ngrex.pattern.NgrexMatch:
        for pattern in self.uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m

    def match_prenegation_uncertainty(self,
                                      graph: nx.DiGraph,
                                      node: bioc.BioCNode) -> ngrex.pattern.NgrexMatch:
        for pattern in self.preneg_uncertain_patterns:
            for m in pattern.finditer(graph):
                n0 = m.group(0)
                if n0 == node:
                    return m


class NegUncDetector(object):
    """
    Detect negation or uncertainty in observation mentions from report(s).

    Original code: https://github.com/stanfordmlgroup/chexpert-labeler/blob/master/stages/classify.py
    """
    def __init__(self, config: CheXpertConfig):
        self.labeler_config = config
        self.parser = parse.NegBioParser(model_dir=self.labeler_config.parsing_model_dir)
        #lemmatizer = lemmatize.Lemmatizer()
        self.ptb2dep = ptb2ud.NegBioPtb2DepConverter(universal=True)

        self.detector = ModifiedDetector(config=self.labeler_config)

    def neg_unc_detect(self, collection: bioc.BioCCollection) -> None:
        """Mark each mention as one of negative, uncertain, or none (positive)."""
        documents = collection.documents
        for document in documents:
            # Parse the report text in place.
            self.parser(document)
            # Add the universal dependency graph in place.
            self.ptb2dep.convert_doc(document)
            # Detect the negation and uncertainty rules in place.
            negdetector = negdetect.NegBioNegDetector(self.detector)
            negdetector(document)
            # To reduce memory consumption, remove sentences text.
            del document.passages[0].sentences[:]
