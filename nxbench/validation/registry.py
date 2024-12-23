"""Integration of validation system with benchmark framework."""

import inspect
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import networkx as nx
import yaml

from nxbench.validation.base import (
    ValidationError,
    validate_communities,
    validate_edge_scores,
    validate_flow,
    validate_node_scores,
    validate_path_lengths,
    validate_scalar_result,
    validate_similarity_scores,
)

warnings.filterwarnings("ignore")

__all__ = [
    "ValidationConfig",
    "ValidationRegistry",
    "BenchmarkValidator",
]

logger = logging.getLogger("nxbench")


@dataclass
class ValidationConfig:
    """Configuration for algorithm result validation."""

    validator: Callable

    params: dict[str, Any] = field(default_factory=dict)

    expected_type: type | None = None

    required: bool = True

    extra_checks: set[str] = field(default_factory=set)


class ValidationRegistry:
    """Registry of validation configurations for algorithms."""

    DEFAULT_VALIDATORS: ClassVar[dict] = {
        "pagerank": ValidationConfig(
            validator=validate_node_scores,
            params={
                "score_range": (0.0, 1.0),
                "require_normalized": False,
                "tolerance": 1e-06,
            },
            expected_type=dict,
        ),
        "betweenness_centrality": ValidationConfig(
            validator=validate_node_scores,
            params={"score_range": (0.0, 1.0), "require_normalized": False},
            expected_type=dict,
        ),
        "eigenvector_centrality": ValidationConfig(
            validator=validate_node_scores,
            params={"score_range": (0.0, 1.0), "require_normalized": False},
            expected_type=dict,
        ),
        "louvain_communities": ValidationConfig(
            validator=validate_communities,
            params={"allow_overlap": False, "check_connectivity": True},
            expected_type=list,
        ),
        "label_propagation_communities": ValidationConfig(
            validator=validate_communities,
            params={"allow_overlap": False, "check_connectivity": False},
            expected_type=list,
        ),
        "shortest_path": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "all_pairs_shortest_path_length": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "maximum_flow": ValidationConfig(
            validator=validate_flow,
            params={"check_conservation": True},
            expected_type=tuple,
        ),
        "jaccard_coefficient": ValidationConfig(
            validator=validate_similarity_scores,
            params={"score_range": (0.0, 1.0), "require_symmetric": True},
            expected_type=list,  # actually an iterator, but we convert it
        ),
        "edge_betweenness_centrality": ValidationConfig(
            validator=validate_edge_scores,
            params={"score_range": (0.0, 1.0)},
            expected_type=dict,
        ),
        "average_clustering": ValidationConfig(
            validator=validate_scalar_result,
            params={"min_value": 0.0, "max_value": 1.0},
            expected_type=float,
        ),
        "local_efficiency": ValidationConfig(
            validator=validate_scalar_result,
            params={"min_value": 0.0, "max_value": 1.0},
            expected_type=float,
        ),
        "number_of_isolates": ValidationConfig(
            validator=validate_scalar_result,
            params={"min_value": 0},
            expected_type=int,
            required=True,
        ),
        "all_pairs_all_shortest_paths": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "all_pairs_dijkstra": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "all_pairs_dijkstra_path_length": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "all_pairs_bellman_ford_path_length": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "all_pairs_bellman_ford_path": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "johnson": ValidationConfig(
            validator=validate_path_lengths,
            params={"check_symmetry": False, "allow_infinity": True},
            expected_type=dict,
        ),
        "closeness_vitality": ValidationConfig(
            validator=validate_node_scores,
            params={"score_range": (0.0, 1.0), "require_normalized": False},
            expected_type=dict,
        ),
    }

    def __init__(self):
        """Initialize the registry with default validators."""
        self._validators: dict[str, ValidationConfig] = self.DEFAULT_VALIDATORS.copy()
        self._custom_validators: dict[str, ValidationConfig] = {}

    def register_validator(
        self, algorithm_name: str, validator: Callable | ValidationConfig, **kwargs
    ) -> None:
        """Register a new validator for an algorithm.

        Parameters
        ----------
        algorithm_name : str
            Name of algorithm to validate
        validator : callable or ValidationConfig
            Validation function or config
        **kwargs : dict
            Additional parameters for ValidationConfig if validator is a callable

        Raises
        ------
        ValueError
            If validator configuration is invalid
        """
        if isinstance(validator, ValidationConfig):
            config = validator
        else:
            if not callable(validator):
                raise TypeError(f"Validator must be callable, got {type(validator)}")

            config = ValidationConfig(validator=validator, **kwargs)

        if not callable(config.validator):
            raise TypeError(f"Invalid validator function: {config.validator}")

        sig = inspect.signature(config.validator)
        if len(sig.parameters) < 2:
            raise ValueError(
                f"Validator must accept at least 2 parameters (result, graph), "
                f"got {len(sig.parameters)}"
            )

        self._custom_validators[algorithm_name] = config
        logger.debug(f"Registered validator for algorithm: {algorithm_name}")

    def get_validator(
        self, algorithm_name: str, *, required: bool = True
    ) -> ValidationConfig | None:
        """Get validator configuration for an algorithm.

        Parameters
        ----------
        algorithm_name : str
            Name of algorithm
        required : bool, default=True
            Whether to raise error if no validator found

        Returns
        -------
        ValidationConfig or None
            Validator configuration if found

        Raises
        ------
        ValueError
            If required=True and no validator found
        """
        config = self._custom_validators.get(algorithm_name)
        if config is not None:
            return config

        config = self._validators.get(algorithm_name)
        if config is not None:
            return config

        if required:
            raise ValueError(f"No validator found for algorithm: {algorithm_name}")

        return None

    def load_config(self, path: str | Path) -> None:
        """Load validator configurations from YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Validator config not found: {path}")

        with path.open() as f:
            config = yaml.safe_load(f)

        for algo_name, validator_config in config.get("validators", {}).items():
            validator_name = validator_config["validator"]
            if hasattr(nx, validator_name):
                validator_func = getattr(nx, validator_name)
            else:
                try:
                    validator_func = globals()[validator_name]
                except KeyError:
                    raise ValueError(f"Unknown validator function: {validator_name}")

            self.register_validator(
                algorithm_name=algo_name,
                validator=validator_func,
                params=validator_config.get("params", {}),
                expected_type=validator_config.get("expected_type"),
                required=validator_config.get("required", True),
                extra_checks=set(validator_config.get("extra_checks", [])),
            )


class BenchmarkValidator:
    """Handles validation for benchmark results."""

    def __init__(self, registry: ValidationRegistry | None = None):
        """Initialize validator with optional registry."""
        self.registry = registry or ValidationRegistry()

    @staticmethod
    def _validate_type(result, expected_type):
        if not isinstance(result, expected_type):
            try:
                result = dict(result)
            except Exception:
                raise TypeError(
                    f"Expected result type {expected_type}, got {type(result)}"
                )
            finally:
                if not isinstance(result, expected_type):
                    raise TypeError(
                        f"Expected result type {expected_type}, got {type(result)}"
                    )

    def validate_result(
        self,
        result: Any,
        algorithm_name: str,
        graph: nx.Graph | nx.DiGraph,
        *,
        raise_errors: bool = True,
    ) -> bool:
        """Validate algorithm result."""
        try:
            config = self.registry.get_validator(algorithm_name, required=False)
            if config is None:
                logger.warning(f"No validator found for algorithm: {algorithm_name}")
                return True

            if config.expected_type:
                self._validate_type(result, config.expected_type)

            config.validator(result, graph, **config.params)

            logger.debug(f"Validation passed for algorithm: {algorithm_name}")
        except Exception as e:
            if raise_errors:
                raise ValidationError(
                    f"Validation failed for {algorithm_name}: {e}"
                ) from e

            logger.exception(
                f"Validation failed for {algorithm_name}: {e}"  # noqa: TRY401
            )
            return False
        else:
            return True

    def create_validator(
        self, algorithm_name: str, *, raise_errors: bool = True
    ) -> Callable[[Any, nx.Graph | nx.DiGraph], bool]:
        """Create a validator function for use with pytest.mark.benchmark."""

        def validator(benchmark_result: Any, graph: nx.Graph | nx.DiGraph) -> bool:
            return self.validate_result(
                benchmark_result, algorithm_name, graph, raise_errors=raise_errors
            )

        return validator


default_validator = BenchmarkValidator()
