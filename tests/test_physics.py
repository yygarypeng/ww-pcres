import numpy as np

from physics.physics import W_MASS, four_vector_pairs, invariant_mass, invariant_mass2
from physics.selection import lepton_p4


def test_invariant_mass_keeps_spacelike_four_vectors_real():
    timelike = np.array([0.0, 0.0, 30.0, np.sqrt(30.0**2 + W_MASS**2)])
    spacelike = np.array([0.0, 0.0, 30.0, 20.0])

    masses = invariant_mass(np.stack([timelike, spacelike]))

    np.testing.assert_allclose(masses[0], W_MASS, rtol=1e-6)
    np.testing.assert_allclose(masses[1] ** 2, np.abs(invariant_mass2(spacelike)), rtol=1e-6)


def test_four_vector_pairs_splits_a_block_into_two_four_vectors():
    block = np.arange(16.0).reshape(2, 8)

    pairs = four_vector_pairs(block)

    assert pairs.shape == (2, 2, 4)
    np.testing.assert_allclose(pairs[1, 0], [8.0, 9.0, 10.0, 11.0])
    np.testing.assert_allclose(pairs[1, 1], [12.0, 13.0, 14.0, 15.0])


def test_lepton_p4_reads_the_lepton_block_of_the_raw_features():
    features = np.arange(18.0).reshape(1, 18)

    leptons = lepton_p4(features)

    assert leptons.shape == (1, 2, 4)
    np.testing.assert_allclose(leptons[0], features[0, :8].reshape(2, 4))
