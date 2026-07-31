"""
SearchSpace Container & Parser.

Parses YAML definitions into Parameter objects via PARAMETER_REGISTRY.
"""

from typing import Dict, Any
from hpo.registry import PARAMETER_REGISTRY
from hpo.validator import SearchSpaceValidator


class SearchSpace:
    """Container holding instantiated Parameter objects for a search space."""

    def __init__(self, search_space_config: Dict[str, Any]):
        SearchSpaceValidator.validate_search_space_config(search_space_config)
        self.config = search_space_config
        self.parameters = {}

        for name, pcfg in search_space_config.items():
            ptype = str(pcfg.get("type", "float")).lower()
            param_cls = PARAMETER_REGISTRY[ptype]
            self.parameters[name] = param_cls(name, pcfg)

    def sample(self, rng: Any = None) -> Dict[str, Any]:
        """Sample a set of parameters from the search space."""
        sampled = {}
        for name, param in self.parameters.items():
            sampled[name] = param.sample(rng=rng)
        return sampled

    def __len__(self) -> int:
        return len(self.parameters)

    def __getitem__(self, item: str):
        return self.parameters[item]
