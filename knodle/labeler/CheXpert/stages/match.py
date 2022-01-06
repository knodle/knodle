"""Define observation extractor class."""
import re
import itertools
from collections import defaultdict
from tqdm import tqdm
import bioc
from examples.labeler.chexpert.constants.constants import *
from . import z_matrix_fct
from . import get_rule_idx


class Extractor(object):
    """Extract observations from reports."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.observation2mention_phrases = self.load_phrases(MENTION_DATA_DIR, "mention")
        self.observation2unmention_phrases = self.load_phrases(UNMENTION_DATA_DIR, "unmention")
        self.add_unmention_phrases()

    def load_phrases(self, phrases_dir, phrases_type):
        """Read in map from observations to phrases for matching."""
        observation2phrases = defaultdict(list)
        for phrases_path in os.listdir(phrases_dir):
            with open(os.path.join(phrases_dir, phrases_path)) as f:
                for line in f:
                    phrase = line.strip().replace("_", " ")
                    observation = phrases_path.replace("_", " ").title()
                    if line:
                        observation2phrases[observation].append(phrase)

        if self.verbose:
            print(f"Loading {phrases_type} phrases for "
                  f"{len(observation2phrases)} observations.")

        return observation2phrases

    def add_unmention_phrases(self):
        """This function is specifically designed for the CheXpert rules."""
        cardiomegaly_mentions\
            = self.observation2mention_phrases[CARDIOMEGALY]
        enlarged_cardiom_mentions\
            = self.observation2mention_phrases[ENLARGED_CARDIOMEDIASTINUM]
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

        self.observation2unmention_phrases[CARDIOMEGALY]\
            = cardiomegaly_unmentions
        self.observation2unmention_phrases[ENLARGED_CARDIOMEDIASTINUM]\
            = enlarged_cardiomediastinum_unmentions

    def overlaps_with_unmention(self, sentence, observation, start, end):
        """Return True if a given match overlaps with an unmention phrase."""
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

    def add_match(self, section, sentence, ann_index, phrase,
                  observation, start, end):
        """Add the match data and metadata to the report object
        in place."""
        annotation = bioc.BioCAnnotation()
        annotation.id = ann_index
        annotation.infons['CUI'] = None
        annotation.infons['semtype'] = None
        annotation.infons['term'] = phrase
        annotation.infons[OBSERVATION] = observation
        annotation.infons['annotator'] = 'Phrase'
        length = end - start
        annotation.add_location(bioc.BioCLocation(sentence.offset + start,
                                                  length))
        annotation.text = sentence.text[start:start+length]

        section.annotations.append(annotation)

    def extract(self, collection):
        """Extract the observations in each report.

        Args:
            collection (BioCCollection): Passages of each report.

        Return:
            extracted_mentions
        """
        self.Z_matrix = z_matrix_fct()

        # The BioCCollection consists of a series of documents.
        # Each document is a report
        documents = collection.documents
        if self.verbose:
            print("Extracting mentions...")
            documents = tqdm(documents)
        for i, document in enumerate(documents):  # added enumerate
            # Get the first section.
            section = document.passages[0]
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

                            self.Z_matrix[i, get_rule_idx(phrase)] = 999  # match, but not clarified if pos/neg/unc
