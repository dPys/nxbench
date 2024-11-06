import logging

import pytest
import yaml

from nxbench.validation.base import ValidationError
from nxbench.validation.registry import (
    BenchmarkValidator,
    ValidationConfig,
    ValidationRegistry,
)


class TestValidationRegistry:
    def test_default_validators_present(self):
        registry = ValidationRegistry()
        assert "pagerank" in registry._validators
        assert "betweenness_centrality" in registry._validators
        assert "louvain_communities" in registry._validators

    def test_register_validator_callable(self, mock_validator):
        registry = ValidationRegistry()
        registry.register_validator("custom_algo", mock_validator)
        assert "custom_algo" in registry._custom_validators
        assert registry._custom_validators["custom_algo"].validator == mock_validator

    def test_register_validator_validation_config(self, mock_validator):
        config = ValidationConfig(
            validator=mock_validator,
            params={"param1": 10},
            expected_type=dict,
            required=True,
            extra_checks={"check1"},
        )
        registry = ValidationRegistry()
        registry.register_validator("custom_algo_config", config)
        assert "custom_algo_config" in registry._custom_validators
        assert registry._custom_validators["custom_algo_config"] == config

    def test_register_validator_invalid_callable(self):
        registry = ValidationRegistry()
        with pytest.raises(ValueError, match=r"Validator must be callable"):
            registry.register_validator("invalid_algo", "not_callable")

    def test_register_validator_non_callable_validation_config(self):
        registry = ValidationRegistry()
        config = ValidationConfig(
            validator="not_callable",
        )
        with pytest.raises(ValueError, match=r"Invalid validator function"):
            registry.register_validator("invalid_config", config)

    def test_register_validator_invalid_signature(self, mock_validator):
        def invalid_validator(result):
            pass

        registry = ValidationRegistry()
        with pytest.raises(
            ValueError, match=r"Validator must accept at least 2 parameters"
        ):
            registry.register_validator("invalid_signature", invalid_validator)

    def test_get_validator_existing(self):
        registry = ValidationRegistry()
        config = registry.get_validator("pagerank")
        assert config is not None
        assert config.validator is not None

    def test_get_validator_non_existing_required(self):
        registry = ValidationRegistry()
        with pytest.raises(
            ValueError, match=r"No validator found for algorithm: unknown_algo"
        ):
            registry.get_validator("unknown_algo")

    def test_get_validator_non_existing_not_required(self):
        registry = ValidationRegistry()
        config = registry.get_validator("unknown_algo", required=False)
        assert config is None

    def test_load_config(self, tmp_path, mock_validator):
        config_data = {
            "validators": {
                "custom_algo_yaml": {
                    "validator": "mock_validator",
                    "params": {"param1": 20},
                    "expected_type": "dict",
                    "required": True,
                    "extra_checks": ["check2"],
                }
            }
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        with pytest.MonkeyPatch.context() as m:
            m.setattr("builtins.globals", lambda: {"mock_validator": mock_validator})
            registry = ValidationRegistry()
            registry.load_config(config_path)

        assert "custom_algo_yaml" in registry._custom_validators
        loaded_config = registry._custom_validators["custom_algo_yaml"]
        assert loaded_config.validator == mock_validator
        assert loaded_config.params == {"param1": 20}
        assert loaded_config.expected_type == "dict"
        assert loaded_config.required is True
        assert loaded_config.extra_checks == {"check2"}

    def test_load_config_unknown_validator_function(self, tmp_path):
        config_data = {
            "validators": {
                "custom_algo_yaml": {
                    "validator": "unknown_validator",
                    "params": {"param1": 20},
                    "expected_type": "dict",
                }
            }
        }
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        registry = ValidationRegistry()
        with pytest.raises(
            ValueError, match=r"Unknown validator function: unknown_validator"
        ):
            registry.load_config(config_path)


class TestBenchmarkValidator:
    def test_validate_result_valid(self, simple_graph):
        result = {node: 1.0 / len(simple_graph) for node in simple_graph.nodes()}
        validator = BenchmarkValidator()
        assert validator.validate_result(result, "pagerank", simple_graph) is True

    def test_validate_result_invalid_type(self, simple_graph):
        result = ["not", "a", "dict"]
        validator = BenchmarkValidator()
        with pytest.raises(
            ValidationError, match=r"Expected result type <class 'dict'>"
        ):
            validator.validate_result(result, "pagerank", simple_graph)

    def test_validate_result_validation_failure(self, simple_graph):
        result = {node: 0.2 for node in simple_graph.nodes()}  # Sum=0.8
        validator = BenchmarkValidator()
        with pytest.raises(
            ValidationError, match=r"Normalized scores sum to .* expected 1\.0"
        ):
            validator.validate_result(result, "pagerank", simple_graph)

    def test_validate_result_no_validator(self, caplog):
        validator = BenchmarkValidator()
        result = 42
        with caplog.at_level(logging.WARNING):
            assert validator.validate_result(result, "unknown_algo", None) is True
            assert "No validator found for algorithm: unknown_algo" in caplog.text

    def test_validate_result_raise_errors_false(self, simple_graph):
        result = {node: 0.2 for node in simple_graph.nodes()}  # Sum=0.8
        validator = BenchmarkValidator()
        assert (
            validator.validate_result(
                result, "pagerank", simple_graph, raise_errors=False
            )
            is False
        )

    def test_create_validator_valid_result(self, simple_graph):
        validator = BenchmarkValidator()
        validate_func = validator.create_validator("pagerank")
        result = {node: 1.0 / len(simple_graph) for node in simple_graph.nodes()}
        validate_func(result, simple_graph)

    def test_create_validator_invalid_result(self, simple_graph):
        validator = BenchmarkValidator()
        validate_func = validator.create_validator("pagerank")
        result = {node: 0.2 for node in simple_graph.nodes()}  # Sum=0.8
        with pytest.raises(
            ValidationError, match=r"Normalized scores sum to .* expected 1\.0"
        ):
            validate_func(result, simple_graph)

    def test_create_validator_no_validator(self, caplog):
        validator = BenchmarkValidator()
        validate_func = validator.create_validator("unknown_algo")
        result = 42
        with caplog.at_level(logging.WARNING):
            assert validate_func(result, None) is True
            assert "No validator found for algorithm: unknown_algo" in caplog.text

    def test_validate_result_with_custom_validator(self, simple_graph, mock_validator):
        registry = ValidationRegistry()
        registry.register_validator("custom_algo", mock_validator)
        validator = BenchmarkValidator(registry)
        result = {"node1": 0.5, "node2": 0.5}
        validator.validate_result(result, "custom_algo", simple_graph)
        mock_validator.assert_called_once_with(result, simple_graph)

    def test_validate_result_with_custom_validator_params(
        self, simple_graph, mock_validator
    ):
        registry = ValidationRegistry()
        config = ValidationConfig(
            validator=mock_validator,
            params={"param1": 10},
            expected_type=dict,
        )
        registry.register_validator("custom_algo_params", config)
        validator = BenchmarkValidator(registry)
        result = {"node1": 0.5, "node2": 0.5}
        validator.validate_result(result, "custom_algo_params", simple_graph)
        mock_validator.assert_called_once_with(result, simple_graph, param1=10)

    def test_validate_result_with_expected_type(self, simple_graph):
        def dummy_validator(result, graph, **params):
            if not isinstance(result, int):
                raise ValidationError("Result is not int")

        registry = ValidationRegistry()
        registry.register_validator(
            "dummy_algo",
            ValidationConfig(
                validator=dummy_validator,
                expected_type=int,
            ),
        )
        benchmark_validator = BenchmarkValidator(registry)
        assert (
            benchmark_validator.validate_result(10, "dummy_algo", simple_graph) is True
        )
        with pytest.raises(
            ValidationError, match=r"Expected result type <class 'int'>"
        ):
            benchmark_validator.validate_result("not int", "dummy_algo", simple_graph)


def test_validation_registry_load_config(temporary_yaml_config, mock_validator):
    registry = ValidationRegistry()
    with pytest.MonkeyPatch.context() as m:
        m.setattr("builtins.globals", lambda: {"mock_validator": mock_validator})
        registry.load_config(temporary_yaml_config)
    assert "custom_algo_yaml" in registry._custom_validators
    config = registry._custom_validators["custom_algo_yaml"]
    assert config.validator == mock_validator
    assert config.params == {"param1": 20}
    assert config.expected_type == "dict"
    assert config.required is True
    assert config.extra_checks == {"check2"}


def test_validation_registry_load_config_missing_file(tmp_path):
    registry = ValidationRegistry()
    config_path = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match=r"Validator config not found"):
        registry.load_config(config_path)


def test_validation_registry_load_config_invalid_validator_name(tmp_path):
    config_data = {
        "validators": {
            "custom_algo_invalid": {
                "validator": "nonexistent_validator",
                "params": {},
            }
        }
    }
    config_path = tmp_path / "config_invalid.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f)
    registry = ValidationRegistry()
    with pytest.raises(
        ValueError, match=r"Unknown validator function: nonexistent_validator"
    ):
        registry.load_config(config_path)


class TestBenchmarkValidatorIntegration:
    def test_validate_result_with_flow(self, flow_graph):
        flow_value = 15.0
        flow_dict = {
            "s": {"A": 10, "B": 5},
            "A": {"B": 0, "t": 10},
            "B": {"t": 5},
            "t": {},
        }
        result = (flow_value, flow_dict)
        validator = BenchmarkValidator()
        assert validator.validate_result(result, "maximum_flow", flow_graph) is True

    def test_validate_result_with_flow_exceeds_capacity(self, flow_graph):
        flow_value = 20.0
        flow_dict = {
            "s": {"A": 15, "B": 5},
            "A": {"t": 10},
            "B": {"t": 5},
            "t": {},
        }
        result = (flow_value, flow_dict)
        validator = BenchmarkValidator()
        with pytest.raises(ValidationError, match=r"exceeds capacity"):
            validator.validate_result(result, "maximum_flow", flow_graph)

    def test_validate_result_with_scalar_result(self):
        validator = BenchmarkValidator()
        result = 3.14
        with pytest.raises(ValidationError, match=r"greater than maximum 1.0"):
            validator.validate_result(result, "local_efficiency", None)

    def test_validate_result_with_scalar_result_invalid(self):
        validator = BenchmarkValidator()
        result = "not a float"
        with pytest.raises(
            ValidationError,
            match=r"Validation failed for local_efficiency: Expected result type <class 'float'>, got <class 'str'>",
        ):
            validator.validate_result(result, "local_efficiency", None)

    def test_validate_result_with_scalar_result_out_of_range(self):
        validator = BenchmarkValidator()
        result = 1.5
        with pytest.raises(
            ValidationError, match=r"Result 1.5 is greater than maximum 1.0"
        ):
            validator.validate_result(result, "local_efficiency", None)

    def test_validate_result_with_number_of_isolates(self):
        validator = BenchmarkValidator()
        result = 2
        assert validator.validate_result(result, "number_of_isolates", None) is True

    def test_validate_result_with_number_of_isolates_invalid(self):
        validator = BenchmarkValidator()
        result = -1
        with pytest.raises(ValidationError, match=r"Result -1 is less than minimum 0"):
            validator.validate_result(result, "number_of_isolates", None)

    def test_create_validator_function(self, simple_graph):
        validator = BenchmarkValidator()
        validate_func = validator.create_validator("pagerank")
        result = {node: 1.0 / len(simple_graph) for node in simple_graph.nodes()}
        validate_func(result, simple_graph)

    def test_create_validator_function_invalid(self, simple_graph):
        validator = BenchmarkValidator()
        validate_func = validator.create_validator("pagerank")
        result = {node: 0.2 for node in simple_graph.nodes()}  # Sum=0.8
        with pytest.raises(
            ValidationError, match=r"Normalized scores sum to .* expected 1\.0"
        ):
            validate_func(result, simple_graph)
