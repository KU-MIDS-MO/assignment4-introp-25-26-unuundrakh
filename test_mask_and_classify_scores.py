import numpy as np
from mask_and_classify_scores import mask_and_classify_scores

def test_cleaning_and_levels_and_pass_counts():
    ## 4x4 test matrix with negatives and values > 100
    arr = np.array([
        [-10,  20,  50, 110],
        [ 40,  69,  70, 101],
        [  0,  39,  41,  80],
        [ 55,  60,  95, 200]
    ])

    cleaned, levels, row_pass_counts = mask_and_classify_scores(arr)

    ##expected cleaned array:
    expected_cleaned = np.array([
        [  0, 20, 50, 100],
        [ 40, 69, 70, 100],
        [  0, 39, 41,  80],
        [ 55, 60, 95, 100]
    ])

    np.testing.assert_array_equal(cleaned, expected_cleaned)

    ##expected levels:
    expected_levels = np.array([
        [0, 0, 1, 2],
        [1, 1, 2, 2],
        [0, 0, 1, 2],
        [1, 1, 2, 2]
    ], dtype=int)

    assert levels.dtype == int
    np.testing.assert_array_equal(levels, expected_levels)

    ##expected passing counts
    expected_pass_counts = np.array([2, 3, 1, 4])
    np.testing.assert_array_equal(row_pass_counts, expected_pass_counts)

def test_no_changes_needed_all_in_range():
    arr = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [ 0, 25, 50, 75],
        [35, 45, 65, 95]
    ])
    cleaned, levels, row_pass_counts = mask_and_classify_scores(arr)

    np.testing.assert_array_equal(cleaned, arr)

    assert levels[0,0] == 0
    assert levels[1,0] == 1
    assert levels[1,2] == 2
    assert levels[3,3] == 2

    expected_pass = np.array([0, 4, 2, 2])
    np.testing.assert_array_equal(row_pass_counts, expected_pass)

def test_invalid_input_types_and_shapes():
    assert mask_and_classify_scores([[1, 2], [3, 4]]) is None

    arr2 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    assert mask_and_classify_scores(arr2) is None

    arr3 = np.array([[1, 2],
                     [3, 4]])
    assert mask_and_classify_scores(arr3) is None

def test_output_structure_minimal_valid_case():
    arr = np.arange(16).reshape(4, 4)
    result = mask_and_classify_scores(arr)

    assert isinstance(result, tuple)
    assert len(result) == 3

    cleaned, levels, row_pass_counts = result
    assert cleaned.shape == arr.shape
    assert levels.shape == arr.shape
    assert row_pass_counts.shape == (4,)
