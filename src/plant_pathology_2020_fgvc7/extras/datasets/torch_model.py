from typing import Any, Dict

import torch

from kedro.io.core import Version
from kedro.extras.datasets.pickle.pickle_dataset import PickleDataSet


def load(fs_file: Any, **kwargs) -> Any:
    import pdb; pdb.set_trace
    return torch.load(fs_file)

def dump(net, fs_file: Any, **kwargs):
    import pdb; pdb.set_trace
    torch.save(net, fs_file)


class TorchModel(PickleDataSet):
    """``TorchModel`` loads / save torch model from a given filepath.

    Example:
    ::

        >>> TorchModel(filepath='resnset.pt')
    """

    def __init__(
        self,
        filepath: str,
        version: Version = None,
        credentials: Dict[str, Any] = None,
    ):
        """Creates a new instance of TorchModel to load / save torch model at the given filepath.

        Args:
            filepath: The location of the torch model file to load / save data.
            version: The version of the dataset being saved and loaded.
            credentials: Credentials required to get access to the underlying filesystem.
        """

        super().__init__(
            filepath=filepath,
            backend="plant_pathology_2020_fgvc7.extras.datasets.torch_model",
            version=version,
            credentials=credentials
        )

    def _load(self) -> Any:
        model = super()._load()
        import pdb; pdb.set_trace
        model.eval()
        return model