"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------

Define observation matcher class.
"""
import bioc
import itertools
import logging
import re
import yaml

from typing import AnyStr, Pattern

from .config import CheXpertConfig
from .utils import z_matrix_fct, get_rule_idx


class Matcher(object):
    """
    Find mentions matching observations in report(s).

    Original code:
    https://github.com/stanfordmlgroup/chexpert-labeler/blob/master/stages/extract.py
    """
    def __init__(self, config: CheXpertConfig, chexpert_data: bool = True):
        self.labeler_config = config

        with open(self.labeler_config.phrases_path) as fp:
            phrases = yaml.load(fp, yaml.FullLoader)

        self.vocab_name = self.labeler_config.phrases_path.stem
        self.observation2mention_phrases = {}
        self.observation2unmention_phrases = {}
        for observation, v in phrases.items():
            if 'include' in v:
                self.observation2mention_phrases[observation] = phrases[observation]['include']
            if 'exclude' in v:
                self.observation2unmention_phrases[observation] = phrases[observation]['exclude']

        logging.debug("Loading mention phrases for %s observations.",
                      len(self.observation2mention_phrases))
        logging.debug("Loading unmention phrases for %s observations.",
                      len(self.observation2unmention_phrases))

        # Add CheXpert specific unmention phrases.
        if chexpert_data:
            self.add_unmention_phrases()

    def add_unmention_phrases(self) -> None:
        """Define additional unmentions. This function is custom-made for the CheXpert rules."""
        cardiomegaly_mentions = self.observation2mention_phrases.get("Cardiomegaly", [])
        enlarged_cardiom_mentions = self.observation2mention_phrases.get("Enlarged Cardiomediastinum", [])
        positional_phrases = (["over the", "overly the", "in the", "assessment of", "diameter of"],
                              ["", " superior", " left", " right"])
        positional_unmentions = \
            [e1 + e2
             for e1 in positional_phrases[0]
             for e2 in positional_phrases[1]]

        cardiomegaly_unmentions = \
            [e1 + " " + e2.replace("the ", "")
             for e1 in positional_unmentions
             for e2 in cardiomegaly_mentions
             if e2 not in ["cardiomegaly", "cardiac enlargement"]]

        enlarged_cardiomediastinum_unmentions = \
            [e1 + " " + e2
             for e1 in positional_unmentions
             for e2 in enlarged_cardiom_mentions]

        self.observation2unmention_phrases[self.labeler_config.cardiomegaly]\
            = cardiomegaly_unmentions
        self.observation2unmention_phrases[self.labeler_config.enlarged_cardiomediastinum]\
            = enlarged_cardiomediastinum_unmentions

    def compile_pattern(self, pattern: AnyStr) -> Pattern[AnyStr]:
        pattern = re.sub(' ', r'\\s+', pattern)
        return re.compile(pattern, re.I | re.M)

    def overlaps_with_unmention(self,
                                sentence: bioc.BioCSentence,
                                observation: str,
                                start: int,
                                end: int) -> bool:
        """Return "True" if a given match overlaps with an unmention phrase."""
        unmention_overlap = False
        unmention_list = self.observation2unmention_phrases.get(observation, [])
        for unmention in unmention_list:
            unmention_pattern = self.compile_pattern(unmention)
            for unmention_match in unmention_pattern.finditer(sentence.text):
                unmention_start, unmention_end = unmention_match.span(0)
                if start < unmention_end and end > unmention_start:
                    unmention_overlap = True
                    return unmention_overlap
            if unmention_overlap:
                return unmention_overlap

        return unmention_overlap

    def match(self, collection: bioc.BioCCollection) -> None:
        """Find the observation matches in each report."""
        # Initialize the Z matrix.
        self.z_matrix = z_matrix_fct(config=self.labeler_config)

        # The BioCCollection consists of a series of documents.
        # Each document is a report.
        documents = collection.documents
        for i, document in enumerate(documents):
            # Get the first (and only) section.
            section = document.passages[0]
            annotation_index = itertools.count(len(section.annotations))

            for sentence in section.sentences:
                obs_phrases = self.observation2mention_phrases.items()
                for observation, phrases in obs_phrases:
                    for phrase in phrases:
                        pattern = self.compile_pattern(phrase)
                        for match in pattern.finditer(sentence.text):
                            start, end = match.span(0)

                            if self.overlaps_with_unmention(sentence,
                                                            observation,
                                                            start,
                                                            end):
                                continue

                            annotation = bioc.BioCAnnotation()
                            annotation.id = str(next(annotation_index))
                            annotation.infons['term'] = phrase
                            annotation.infons["observation"] = observation
                            annotation.infons['annotator'] = 'RegEx'
                            annotation.infons['vocab'] = self.vocab_name
                            annotation.add_location(bioc.BioCLocation(sentence.offset + start,
                                                                      end - start))
                            annotation.text = sentence.text[start:end]
                            section.annotations.append(annotation)

                            # Corresponding Z matrix position is adjusted, indicating that there was a match.
                            # At this point it is not clear though, whether it is positive, negative or uncertain.
                            self.z_matrix[i, get_rule_idx(phrase,
                                                          config=self.labeler_config)] = self.labeler_config.match
