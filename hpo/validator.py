"""
Search Space Validator Utility.

Catches configuration errors, invalid parameter bounds, empty lists, and invalid types
before optimization begins.
"""

from typing import Dict, Any
from hpo.registry import PARAMETER_REGISTRY


class SearchSpaceValidator:
    """Validates HPO configuration dictionary and search space settings."""

    @staticmethod
    def validate_search_space_config(search_space_dict: Dict[str, Any]) -> bool:
        """
        Validate dictionary of search space parameter definitions.

        Args:
            search_space_dict: Dictionary mapping param_name -> config_dict

        Returns:
            True if valid, raises ValueError if invalid
        """
        if not search_space_dict:
            raise ValueError("Search space dictionary is empty.")

        seen_names = set()
        for name, param_cfg in search_space_dict.items():
            if name in seen_names:
                raise ValueError(f"Duplicate parameter name detected: {name}")
            seen_names.add(name)

            if not isinstance(param_cfg, dict):
                raise TypeError(f"Parameter '{name}' definition must be a dictionary.")

            param_type = str(param_cfg.get("type", "float")).lower()
            if param_type not in PARAMETER_REGISTRY:
                raise ValueError(
                    f"Unsupported parameter type '{param_type}' for parameter '{name}'. Supported types: {list(PARAMETER_REGISTRY.keys())}"
                )

            # Instantiate parameter to trigger internal type validation
            param_cls = PARAMETER_REGISTRY[param_type]
            param_obj = param_cls(name, param_cfg)
            param_obj.validate()

        return True
