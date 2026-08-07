"""
Tests that the walk-forward fold boundaries (walk_forward_pipeline.py) are
strictly chronological with no leakage across the train/test boundary of
any fold -- i.e. genuinely walk-forward, never k-fold or randomly split.
"""
import numpy as np
import pytest

from walk_forward_pipeline import build_fold_boundaries, N_FOLDS, INITIAL_TRAIN_FRAC


class TestFoldBoundaries:
    def test_folds_are_expanding_and_contiguous(self):
        n_rows = 3774
        boundaries = build_fold_boundaries(n_rows, n_folds=5, initial_frac=0.55)
        assert len(boundaries) == 5

        prev_train_end = None
        for train_end, test_start, test_end in boundaries:
            # train window always starts at 0 (expanding, not sliding)
            assert test_start == train_end, "test must start exactly where train ends"
            assert test_end > test_start, "test window must be non-empty"
            if prev_train_end is not None:
                assert train_end == prev_train_end or train_end > prev_train_end, \
                    "train window must never shrink between folds"
                assert train_end >= prev_train_end, "expanding window must not shrink"
            prev_train_end = test_end  # next fold's train includes this fold's test block

        # last fold must reach the end of the data
        assert boundaries[-1][2] == n_rows

    def test_no_test_block_precedes_its_own_training_data(self):
        """For every fold, all training indices are strictly less than all test indices."""
        n_rows = 2000
        boundaries = build_fold_boundaries(n_rows, n_folds=4, initial_frac=0.5)
        for train_end, test_start, test_end in boundaries:
            train_indices = set(range(0, train_end))
            test_indices = set(range(test_start, test_end))
            assert train_indices.isdisjoint(test_indices)
            assert max(train_indices, default=-1) < min(test_indices, default=n_rows + 1)

    def test_test_blocks_across_folds_do_not_overlap(self):
        """Each fold's test block must be a distinct, non-overlapping slice of time."""
        n_rows = 3000
        boundaries = build_fold_boundaries(n_rows, n_folds=5, initial_frac=0.5)
        test_ranges = [set(range(s, e)) for _, s, e in boundaries]
        for i in range(len(test_ranges)):
            for j in range(i + 1, len(test_ranges)):
                assert test_ranges[i].isdisjoint(test_ranges[j]), \
                    f"fold {i} and fold {j} test windows overlap -- not a valid walk-forward split"

    def test_folds_cover_the_full_post_initial_period_with_no_gaps(self):
        n_rows = 2500
        boundaries = build_fold_boundaries(n_rows, n_folds=5, initial_frac=0.5)
        initial_end = int(n_rows * 0.5)
        covered = set()
        for _, test_start, test_end in boundaries:
            covered |= set(range(test_start, test_end))
        assert covered == set(range(initial_end, n_rows)), \
            "walk-forward test windows must tile the post-initial-training period with no gaps or overlaps"

    def test_default_config_produces_valid_boundaries(self):
        """Sanity check the actual constants used by the real pipeline run."""
        boundaries = build_fold_boundaries(3774, n_folds=N_FOLDS, initial_frac=INITIAL_TRAIN_FRAC)
        assert len(boundaries) == N_FOLDS
        for train_end, test_start, test_end in boundaries:
            assert 0 < train_end <= 3774
            assert test_start < test_end <= 3774
