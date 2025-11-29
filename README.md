[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tCPpQfCH)
## Introduction to Programming
## Winter Semester 2025/26 -- Assignment 4
## All concepts required for these tasks (Boolean indexing, logical operators, simple loops) have been covered in the lectures and examples.

In this assignment you will write one NumPy function that perform several
steps of analysis on a square array of scores


Input rules:
- arr must be a NumPy array(np.ndarray)
- arr must be 2-dimensional and square (shape n × n)
-  n must be at least 4

If any of these rules are not satisfied,the function should return None

PartA;cleaning scores

Create a modified copy of the input array, called cleaned, where:
- all values less than 0 are replaced with 0
- all values greater than 100 are replaced with 100

cleaned will be the first element of the return value.

Part B; classifying scores

Based on cleaned, create another array with the same shape, called levels.
Each entry in levels is an integer that encodes the “level” of the score:

- 0 for “low” scores       (< 40)
- 1 for “medium” scores    (40 ≤ value < 70)
- 2 for “high” scores      (value ≥ 70)

levels will be the second element of the return value.

Part C; counting passing scores per row

A passing score is any value ≥ 50 (in the cleaned array).

For each row of cleaned, count how many entries are passing scores.
Store these counts in a 1-dimensional NumPy array row_pass_counts of length n.

***Use normal Python loops to do this (you do not need sum with an axis)***

row_pass_counts will be the third element of the return value.

Return value

If the input is valid, the function must return a tuple:

    (cleaned, levels, row_pass_counts)

If the input is invalid, the function must return None.

how to use the files;

you write your solution in;
- mask_and_classify_scores.py

The tests will be run using;
- test_mask_and_classify_scores.py

run tests with;
    pytest test_mask_and_classify_scores.py
