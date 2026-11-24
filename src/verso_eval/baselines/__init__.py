"""Baselines (docs/VERSO_Pipeline_Build_Plan.md, section 5.3).

mllm_zero_shot      MLLM solves the problem from the diagram and the text (section 5.3.1, decisions D28-D32); its
                    request and the extraction of its answer are in verso.direct, which VERSO uses too (D73)
autovsr             AutoVSR's netlist building on the evaluation diagrams, Topology F1 (E-10, EV-08); what the runs of
                    AutoVSR share
autovsr_answers     AutoVSR's whole pipeline on the evaluation questions, Answer Accuracy (E-10, EV-26)
autovsr_reproduce   AutoVSR's whole pipeline on a sample of its paper's data, against its Table 1 (E-23, EV-19)
"""
