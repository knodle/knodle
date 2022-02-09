import os
import tempfile

from bllipparser import ModelFetcher


class WeatherConfig:
    def __init__(
            self,
            parsing_model_dir: str = None,

            chexpert_data_dir: str = os.path.join(os.getcwd(), "examples", "labeler", "chexpert"),

            observation: str = "observation",

            # Numeric constants
            match: int = 999,  # choose any number but 1, 0 or -1
            positive: int = 1,
            negative: int = 0,
            uncertain: int = -1,

            # Misc. constants
            uncertainty: str = "uncertainty",
            negation: str = "negation",
            reports: str = "Reports"
    ):
        self.parsing_model_dir = os.path.expanduser(ModelFetcher.download_and_install_model(
            'GENIA+PubMed', os.path.join(tempfile.gettempdir(), 'models'))) if parsing_model_dir is None \
            else parsing_model_dir

        self.mention_data_dir = os.path.join(chexpert_data_dir, "phrases", "mention")
        self.unmention_data_dir = os.path.join(chexpert_data_dir, "phrases", "unmention")

        self.pre_neg_unc_path = os.path.join(chexpert_data_dir, "patterns", "pre_negation_uncertainty.txt")
        self.neg_path = os.path.join(chexpert_data_dir, "patterns", "negation.txt")
        self.post_neg_unc_path = os.path.join(chexpert_data_dir, "patterns", "post_negation_uncertainty.txt")

        self.sample_path = os.path.join(chexpert_data_dir, "reports", "weather_forecast.csv")

        self.output_dir = os.path.join(chexpert_data_dir, "output")

        # Get all the mention files and sort them alphabetically to avoid undesired behaviour when T-matrix is created
        self.files = os.listdir(self.mention_data_dir)
        self.files.sort()

        self.observation = observation

        # Numeric constants
        self.match = match
        self.positive = positive
        self.negative = negative
        self.uncertain = uncertain

        # Misc. constants
        self.uncertainty = uncertainty
        self.negation = negation
        self.reports = reports
