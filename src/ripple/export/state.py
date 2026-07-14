"""Explicit, deterministic flattening for streaming model state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class StateTensorSpec:
    """Portable description of one state tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"
    reset_policy: str = "hard"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "reset_policy": self.reset_policy,
        }


@dataclass(frozen=True)
class _TreeNode:
    kind: str
    children: tuple[_TreeNode, ...] = ()
    keys: tuple[str, ...] = ()
    python_type: type[Any] | None = None
    leaf_index: int | None = None


@dataclass(frozen=True)
class FlattenedState:
    """Flat tensor values plus enough static structure to rebuild state."""

    tensors: tuple[Any, ...]
    tensor_specs: tuple[StateTensorSpec, ...]
    tree: _TreeNode

    def manifest_specs(self) -> list[dict[str, object]]:
        return [spec.to_dict() for spec in self.tensor_specs]


def _is_tensor(value: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "detach")
    )


def _tensor_name(path: Sequence[str]) -> str:
    raw = "state_" + "_".join(path or ("root",))
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)


def flatten_state(state: Any) -> FlattenedState:
    """Flatten tensor-only nested state with stable mapping key ordering.

    Mappings, lists, tuples, and dataclass instances are accepted. Mapping keys
    must be strings. Scalars and other hidden Python state are rejected.
    """

    tensors: list[Any] = []
    specs: list[StateTensorSpec] = []

    def visit(value: Any, path: tuple[str, ...]) -> _TreeNode:
        if _is_tensor(value):
            index = len(tensors)
            name = _tensor_name(path)
            tensors.append(value)
            specs.append(
                StateTensorSpec(
                    name=name,
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(value.dtype).removeprefix("torch."),
                )
            )
            return _TreeNode(kind="tensor", leaf_index=index)

        if is_dataclass(value) and not isinstance(value, type):
            field_names = tuple(field.name for field in fields(value))
            children = tuple(
                visit(getattr(value, name), (*path, name)) for name in field_names
            )
            return _TreeNode(
                kind="dataclass",
                children=children,
                keys=field_names,
                python_type=type(value),
            )

        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("state mapping keys must be strings")
            keys = tuple(sorted(value))
            children = tuple(visit(value[key], (*path, key)) for key in keys)
            return _TreeNode(kind="mapping", children=children, keys=keys)

        if isinstance(value, tuple):
            children = tuple(
                visit(item, (*path, str(index))) for index, item in enumerate(value)
            )
            return _TreeNode(
                kind="tuple", children=children, python_type=type(value)
            )

        if isinstance(value, list):
            children = tuple(
                visit(item, (*path, str(index))) for index, item in enumerate(value)
            )
            return _TreeNode(kind="list", children=children)

        raise TypeError(
            f"streaming state must contain tensors only; found {type(value).__name__}"
        )

    tree = visit(state, ())
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("state paths produce duplicate portable tensor names")
    return FlattenedState(tuple(tensors), tuple(specs), tree)


def unflatten_state(tensors: Sequence[Any], tree: _TreeNode) -> Any:
    """Rebuild a state value using a tree returned by :func:`flatten_state`."""

    def visit(node: _TreeNode) -> Any:
        if node.kind == "tensor":
            assert node.leaf_index is not None
            try:
                return tensors[node.leaf_index]
            except IndexError as error:
                raise ValueError("not enough state tensors for tree") from error
        values = [visit(child) for child in node.children]
        if node.kind == "mapping":
            return dict(zip(node.keys, values, strict=True))
        if node.kind == "list":
            return values
        if node.kind == "tuple":
            tuple_type = node.python_type or tuple
            if hasattr(tuple_type, "_fields"):
                return tuple_type(*values)
            return tuple_type(values)
        if node.kind == "dataclass":
            assert node.python_type is not None
            return node.python_type(**dict(zip(node.keys, values, strict=True)))
        raise ValueError(f"unknown state tree node kind: {node.kind}")

    rebuilt = visit(tree)
    expected = _maximum_leaf_index(tree) + 1
    if len(tensors) != expected:
        raise ValueError(f"expected {expected} state tensors, got {len(tensors)}")
    return rebuilt


def flatten_state_values(state: Any, tree: _TreeNode) -> tuple[Any, ...]:
    """Flatten values using an already validated static tree.

    This trace-friendly variant deliberately performs no tensor introspection.
    """

    values: list[Any] = []

    def visit(value: Any, node: _TreeNode) -> None:
        if node.kind == "tensor":
            values.append(value)
            return
        if node.kind in ("mapping", "dataclass"):
            for key, child in zip(node.keys, node.children, strict=True):
                item = value[key] if node.kind == "mapping" else getattr(value, key)
                visit(item, child)
            return
        if node.kind in ("list", "tuple"):
            for index, child in enumerate(node.children):
                visit(value[index], child)
            return
        raise ValueError(f"unknown state tree node kind: {node.kind}")

    visit(state, tree)
    return tuple(values)


def _maximum_leaf_index(node: _TreeNode) -> int:
    if node.kind == "tensor":
        assert node.leaf_index is not None
        return node.leaf_index
    if not node.children:
        return -1
    return max(_maximum_leaf_index(child) for child in node.children)


def validate_flat_state(
    tensors: Sequence[Any], specs: Sequence[StateTensorSpec]
) -> None:
    """Check state count, fixed shapes, and dtypes against a schema."""

    if len(tensors) != len(specs):
        raise ValueError(f"expected {len(specs)} state tensors, got {len(tensors)}")
    for tensor, spec in zip(tensors, specs, strict=True):
        shape = tuple(int(dim) for dim in tensor.shape)
        dtype = str(tensor.dtype).removeprefix("torch.")
        if shape != spec.shape:
            raise ValueError(f"{spec.name}: expected shape {spec.shape}, got {shape}")
        if dtype != spec.dtype:
            raise ValueError(f"{spec.name}: expected dtype {spec.dtype}, got {dtype}")
