"""Parsers for the compact MoveApps settings."""

from __future__ import annotations

import math
import re

from hmmcma import CriterionConfig, Feature, SegmentationRule


FEATURES_BY_NUMBER = {
    1: Feature.SPEED,
    2: Feature.DISTANCE,
    3: Feature.TURN_ANGLE,
    4: Feature.DIRECTION,
    5: Feature.ANGULAR_DIFFERENCE,
    6: Feature.PERSISTENCE_VELOCITY,
    7: Feature.TERRAIN,
}

_CRITERION_FACTORIES = {
    "minimum_hops": (CriterionConfig.minimum_hops, 1),
    "maximum_hops": (CriterionConfig.maximum_hops, 1),
    "minimum_duration": (CriterionConfig.minimum_duration, 1),
    "speed": (CriterionConfig.speed, 1),
    "speed_with_outliers": (CriterionConfig.speed_with_outliers, 2),
    "turning_angle": (CriterionConfig.turning_angle, 1),
    "turn_angle": (CriterionConfig.turning_angle, 1),
    "turning_angle_with_outliers": (CriterionConfig.turning_angle_with_outliers, 2),
    "heading": (CriterionConfig.heading, 1),
    "heading_with_outliers": (CriterionConfig.heading_with_outliers, 2),
    "disk": (CriterionConfig.disk, 1),
    "disc": (CriterionConfig.disk, 1),
}


def parse_features(value) -> tuple[Feature, ...]:
    """Parse a selection such as ``1,3`` into unique hmmcma features."""
    tokens = value if isinstance(value, (list, tuple)) else re.split(r"[\s,]+", str(value).strip())
    if not tokens or all(str(token).strip() == "" for token in tokens):
        raise ValueError("Select at least one HMM feature (for example '1,3').")

    selected = []
    for token in tokens:
        if str(token).strip() == "":
            continue
        try:
            number = int(token)
            feature = FEATURES_BY_NUMBER[number]
        except (TypeError, ValueError, KeyError) as exc:
            choices = ", ".join(f"{number}={feature.value}" for number, feature in FEATURES_BY_NUMBER.items())
            raise ValueError(f"Unknown feature number {token!r}. Choose from: {choices}.") from exc
        if feature not in selected:
            selected.append(feature)
    return tuple(selected)


def parse_rules(value, num_states: int) -> tuple[SegmentationRule, ...]:
    """Parse ``label:criterion=value; ...`` into C++-backed rules."""
    definitions = [definition.strip() for definition in str(value).split(";") if definition.strip()]
    if len(definitions) != num_states:
        raise ValueError(
            f"criteria_rules must contain exactly {num_states} semicolon-separated rules; "
            f"received {len(definitions)}."
        )

    rules = []
    for index, definition in enumerate(definitions, start=1):
        if ":" in definition:
            label, expression = (part.strip() for part in definition.split(":", 1))
        else:
            label, expression = f"state_{index}", definition
        if not label or not expression:
            raise ValueError(f"Invalid criteria rule {index}: both label and expression are required.")
        rules.append(SegmentationRule(label, _parse_expression(expression, index)))

    labels = [rule.label for rule in rules]
    if len(set(labels)) != len(labels):
        raise ValueError("Criteria rule labels must be unique.")
    return tuple(rules)


def _parse_expression(expression: str, rule_number: int):
    has_and = "&" in expression
    has_or = "|" in expression
    if has_and and has_or:
        raise ValueError(f"Criteria rule {rule_number} mixes '&' and '|'. Use only one operator per rule.")
    separator = "&" if has_and else "|" if has_or else None
    atom_texts = [part.strip() for part in expression.split(separator)] if separator else [expression.strip()]
    if any(not atom for atom in atom_texts):
        raise ValueError(f"Criteria rule {rule_number} contains an empty criterion.")
    criteria = [_parse_atom(atom, rule_number) for atom in atom_texts]
    if len(criteria) == 1:
        return criteria[0]
    return CriterionConfig.all_of(criteria) if separator == "&" else CriterionConfig.any_of(criteria)


def _parse_atom(atom: str, rule_number: int):
    negate = atom.startswith("!")
    if negate:
        atom = atom[1:].strip()
    if "=" not in atom:
        raise ValueError(f"Criteria rule {rule_number}: expected 'criterion=value', got {atom!r}.")
    name, raw_values = (part.strip().lower() for part in atom.split("=", 1))
    factory_spec = _CRITERION_FACTORIES.get(name)
    if factory_spec is None:
        choices = ", ".join(sorted(_CRITERION_FACTORIES))
        raise ValueError(f"Criteria rule {rule_number}: unknown criterion {name!r}. Choose from: {choices}.")

    factory, argument_count = factory_spec
    values = [part.strip() for part in raw_values.split(",")]
    if len(values) != argument_count:
        raise ValueError(f"Criteria rule {rule_number}: {name} expects {argument_count} numeric value(s).")
    try:
        arguments = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"Criteria rule {rule_number}: {name} values must be numeric.") from exc
    if any(not math.isfinite(value) or value < 0 for value in arguments):
        raise ValueError(f"Criteria rule {rule_number}: {name} values must be finite and non-negative.")

    if name in {"minimum_hops", "maximum_hops"} or name.endswith("_with_outliers"):
        integer_positions = range(len(arguments)) if name in {"minimum_hops", "maximum_hops"} else (1,)
        if any(not arguments[position].is_integer() for position in integer_positions):
            raise ValueError(f"Criteria rule {rule_number}: hop and outlier counts must be integers.")
        for position in integer_positions:
            arguments[position] = int(arguments[position])

    criterion = factory(*arguments)
    return CriterionConfig.negate(criterion) if negate else criterion
