# Conditional Angular And Kinematic MMD Design

## Goal

Regularize decay angles and charge-associated neutrino/W kinematics with two
physically grouped conditional MMD losses while retaining event-wise W
four-vector supervision.

## Physical Definitions

Use the dataset's fixed charge ordering throughout: slot 0 is associated with
the positive lepton and slot 1 with the negative lepton. Do not infer or sort
on-shell and off-shell assignments.

For truth and prediction, reconstruct the charge-associated neutrinos as

`nu_pos = w_pos - lep_pos` and `nu_neg = w_neg - lep_neg`.

Define the positive-associated momentum fraction from spatial magnitudes:

`alpha_pos = |p_nu_pos| / (|p_nu_pos| + |p_nu_neg|)`.

Use `0.5` only when the total magnitude is numerically zero, where the fraction
is undefined. Only `alpha_pos` is needed because the complementary fraction is
`1 - alpha_pos`.

## Conditional MMD Features

Keep the existing angular feature:

1. `theta_pos / pi`
2. `sin(phi_pos)`
3. `cos(phi_pos)`
4. `theta_neg / pi`
5. `sin(phi_neg)`
6. `cos(phi_neg)`

Define a separate kinematic feature:

1. `2 * alpha_pos - 1`
2. `m2_w_pos / W_MASS_SCALE**2`
3. `m2_w_neg / W_MASS_SCALE**2`

Mass-squared is used directly so training does not require a square root of a
potentially non-positive prediction. The fixed physical normalization puts the
mass-squared values on a scale comparable to the centered fraction. Additional
training-sample mean/std standardization is not required.

## Losses

Compare truth and prediction separately for the six-component angular feature
and the three-component kinematic feature. Both losses use the existing
normalized conditioning representation of `m_ll`, `deta_ll`, `dphi_ll`, and
`dphi_llmet`.

The training objective retains:

- Event-wise standardized W four-vector Huber loss.
- Conditional angular MMD for the joint positive/negative decay-angle
  distribution.
- Conditional kinematic MMD for the joint momentum-sharing and W-virtuality
  distribution.
- Optional event-wise W-mass Huber loss.

Replace the global W-mass MMD with the conditional kinematic MMD; do not train
both constraints. Do not add a separate alpha loss because alpha and both W
masses are already calibrated together by the kinematic MMD.

## Validity And Diagnostics

Build angular features on the common mask where both truth and predicted W
rest-frame calculations are valid. The kinematic loss uses all rows where its
truth, prediction, and condition features are finite. If no events remain for
either loss, return a differentiable zero.

Log both conditional losses independently and retain gradient-cosine logging
against the main loss. Evaluate standalone alpha and W-mass distributions plus
key alpha-angle and mass-angle correlations during validation. The separate
losses constrain correlations within each feature group and with the observed
condition, but not residual correlations between the two groups.

## Testing

Add focused tests for charge association, alpha edge cases, kinematic feature
order and scaling, angular masking, finite-row handling, and propagation of
the normalized condition to both local MMD losses. Existing W-mass Huber and
angular behavior must remain covered.

## Non-Goals

- Resolving the intrinsically underconstrained event kinematics.
- Changing model outputs, charge ordering, or the W reconstruction layer.
- Retuning loss weights or MMD bandwidths in the initial implementation.
- Directly constraining residual angle-kinematic correlations beyond their
  shared dependence on the conditioning observables.
