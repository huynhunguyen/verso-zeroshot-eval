"""Evaluation side of VERSO: gold data, check questions, grading and experiment runs.

Only this package reads gold data (gold netlists, check questions, dataset answers). The pipeline package
`verso` never imports it; tests/test_boundaries.py enforces that.
"""
