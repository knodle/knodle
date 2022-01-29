"""
This code builds upon the CheXpert labeler from Stanford ML Group.
It has been slightly modified to be compatible with knodle.
The original code can be found here: https://github.com/stanfordmlgroup/chexpert-labeler

----------------------------------------------------------------------------------------

Define observation matcher class.
"""
import bioc
import itertools
import os
import re
from collections import defaultdict
from typing import DefaultDict

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
        self.observation2mention_phrases = self.load_phrases(self.labeler_config.mention_data_dir)
        self.observation2unmention_phrases = self.load_phrases(self.labeler_config.unmention_data_dir)

        # Add CheXpert specific unmention phrases.
        if chexpert_data:
            self.add_unmention_phrases()

    def load_phrases(self, phrases_dir: str) -> DefaultDict:
        """Read in map from observations to phrases for matching."""
        observation2phrases = defaultdict(list)
        for phrases_path in os.listdir(phrases_dir):
            with open(os.path.join(phrases_dir, phrases_path)) as f:
                for line in f:
                    phrase = line.strip().replace("_", " ")
                    observation = phrases_path.replace("_", " ").title()
                    if line:
                        observation2phrases[observation].append(phrase)

        return observation2phrases

    def add_unmention_phrases(self) -> None:
        """Define additional unmentions. This function is custom-made for the CheXpert rules."""
        cardiomegaly_mentions\
            = self.observation2mention_phrases[self.labeler_config.cardiomegaly]
        enlarged_cardiom_mentions\
            = self.observation2mention_phrases[self.labeler_config.enlarged_cardiomediastinum]
        positional_phrases = (["over the", "overly the", "in the"],
                              ["", " superior", " left", " right"])
        positional_unmentions\
            = [e1 + e2
               for e1 in positional_phrases[0]
               for e2 in positional_phrases[1]]
        cardiomegaly_unmentions\
            = [e1 + " " + e2.replace("the ", "")
               for e1 in positional_unmentions
               for e2 in cardiomegaly_mentions
               if e2 not in ["cardiomegaly",
                             "cardiac enlargement"]]
        enlarged_cardiomediastinum_unmentions\
            = [e1 + " " + e2
               for e1 in positional_unmentions
               for e2 in enlarged_cardiom_mentions]

        self.observation2unmention_phrases[self.labeler_config.cardiomegaly]\
            = cardiomegaly_unmentions
        self.observation2unmention_phrases[self.labeler_config.enlarged_cardiomediastinum]\
            = enlarged_cardiomediastinum_unmentions

    def overlaps_with_unmention(self,
                                sentence: bioc.BioCSentence,
                                observation: str,
                                start: int,
                                end: int) -> bool:
        """Return "True" if a given match overlaps with an unmention phrase."""
        unmention_overlap = False
        unmention_list = self.observation2unmention_phrases.get(observation,
                                                                [])
        for unmention in unmention_list:
            unmention_matches = re.finditer(unmention, sentence.text)
            for unmention_match in unmention_matches:
                unmention_start, unmention_end = unmention_match.span(0)
                if start < unmention_end and end > unmention_start:
                    unmention_overlap = True
                    break  # break early if overlap is found
            if unmention_overlap:
                break  # break early if overlap is found

        return unmention_overlap

    def add_match(self,
                  section: bioc.BioCPassage,
                  sentence: bioc.BioCSentence,
                  ann_index: str,
                  phrase: str,
                  observation: str,
                  start: int,
                  end: int) -> None:
        """Add the match data and metadata to the report object in place."""
        annotation = bioc.BioCAnnotation()
        annotation.id = ann_index
        annotation.infons['CUI'] = None  # TODO: check if necessary
        annotation.infons['semtype'] = None
        annotation.infons['term'] = phrase
        annotation.infons[self.labeler_config.observation] = observation
        annotation.infons['annotator'] = 'Phrase'
        length = end - start
        annotation.add_location(bioc.BioCLocation(sentence.offset + start,
                                                  length))
        annotation.text = sentence.text[start:start+length]

        section.annotations.append(annotation)

    def match(self, collection: bioc.BioCCollection) -> None:
        """Find the observation matches in each report."""
        # Initialize the Z matrix.
        self.z_matrix = z_matrix_fct(config=self.labeler_config)

        # The BioCCollection consists of a series of documents.
        # Each document is a report.
        documents = collection.documents
        for i, document in enumerate(documents):
            # Get the first section.
            section = document.passages[0]  # TODO: check if necessary
            annotation_index = itertools.count(len(section.annotations))

            for sentence in section.sentences:
                obs_phrases = self.observation2mention_phrases.items()
                for observation, phrases in obs_phrases:
                    for phrase in phrases:
                        matches = re.finditer(phrase, sentence.text)

                        for match in matches:
                            start, end = match.span(0)

                            if self.overlaps_with_unmention(sentence,
                                                            observation,
                                                            start,
                                                            end):
                                continue

                            self.add_match(section,
                                           sentence,
                                           str(next(annotation_index)),
                                           phrase,
                                           observation,
                                           start,
                                           end)

                            # Corresponding Z matrix position is adjusted, indicating that there was a match.
                            # At this point it is not clear though, whether it is positive, negative or uncertain.
                            self.z_matrix[i, get_rule_idx(phrase,
                                                          config=self.labeler_config)] = self.labeler_config.match
