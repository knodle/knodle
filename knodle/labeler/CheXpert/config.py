from pathlib import Path
import os


class ChexpertConfig:
    def __init__(
            self,
            home_dir: str = Path.home(),
            chexpert_data_dir: str = os.path.join(os.getcwd(), "examples", "labeler", "chexpert"),

            # Observation constants - CheXpert data specific
            cardiomegaly: str = "Cardiomegaly",
            enlarged_cardiomediastinum: str = "Enlarged Cardiomediastinum",
            support_devices: str = "Support Devices",
            no_finding: str = "No Finding",
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

        self.parsing_model_dir = home_dir / ".local/share/bllipparser/GENIA+PubMed"

        self.mention_data_dir = os.path.join(chexpert_data_dir, "phrases", "mention")
        self.unmention_data_dir = os.path.join(chexpert_data_dir, "phrases", "unmention")

        self.pre_neg_unc_path = os.path.join(chexpert_data_dir, "patterns", "pre_negation_uncertainty.txt")
        self.neg_path = os.path.join(chexpert_data_dir, "patterns", "negation.txt")
        self.post_neg_unc_path = os.path.join(chexpert_data_dir, "patterns", "post_negation_uncertainty.txt")

        self.sample_path = os.path.join(chexpert_data_dir, "reports", "sample_reports.csv")

        self.output_dir = os.path.join(chexpert_data_dir, "output")

        # Get all the mention files and sort them alphabetically to avoid undesired behaviour when T-matrix is created
        self.files = os.listdir(self.mention_data_dir)
        self.files.sort()

        # Observation constants - CheXpert data specific
        self.cardiomegaly = cardiomegaly
        self.enlarged_cardiomediastinum = enlarged_cardiomediastinum
        self.support_devices = support_devices
        self.no_finding = no_finding
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
