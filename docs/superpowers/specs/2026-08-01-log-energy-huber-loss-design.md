# Log-Energy Huber Loss Design

## Goal

Compute the standardized four-vector Huber loss in a space where each W
boson's energy is transformed with `log1p`, while momentum components remain
linear.

## Design

Treat the first eight target and prediction values as two four-vectors with
layout `[px, py, pz, E]`. Reshape them to `[..., 2, 4]`, apply `log1p` to the
energy component of both vectors, and flatten them back before computing the
standardized residual and mean Huber loss.

Compute the four shared component scales from training targets after applying
the same `log1p` energy transformation. This keeps the energy residual and its
scale in the same transformed space. Continue clamping scales to a positive
floating-point epsilon.

The current data validation and physics-aware output layer guarantee positive
energies. `log1p` also keeps zero energy finite, but inputs less than or equal
to `-1` remain invalid.

## Testing

Add focused tests that verify:

- both W energy slots use `log1p` while momentum slots remain unchanged;
- the shared energy scale is computed from `log1p(E)`;
- zero energies produce a finite loss.

Run the focused model-loss tests, followed by the full test suite if the
focused tests pass.
