from gpflow.monitor import ModelToTensorBoard
from gpflow.models import BayesianModel
from typing import List
from gpflow.utilities import parameter_dict


class CustomModelToTensorBoard(ModelToTensorBoard):
    def __init__(
        self,
        log_dir: str,
        model: BayesianModel
    ):
        super().__init__(log_dir, model, max_size=8)

    def run(self, **unused_kwargs):
        for name, parameter in parameter_dict(self.model).items():
            if 'layers' not in name and ('variance' in name or 'lengthscales' in name or 'angle' in name or 'lims' in name) or 'theta' in name:  # "layers" filters out neural net layers
                if 'kernel' in name:  # process kernel naming 
                    name = name.split('.')[-1]
                    name = 'kernel.' + name
                # also add likelihood params
                self._summarize_parameter(name, parameter)
