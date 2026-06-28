# Article and IEEE_TSP Synchronization Rule

This project maintains two manuscript surfaces:

- `article/`: JISA/Elsevier CAS source with modular section files.
- `IEEE_TSP/`: IEEE Transactions on Signal Processing source with a monolithic `main.tex`.

Synchronization rule:

1. Any scientific text, table, result, figure caption, theorem, limitation, declaration, or reference change made in `article/` must be propagated to `IEEE_TSP/`.
2. Any scientific text, table, result, figure caption, theorem, limitation, declaration, or reference change made in `IEEE_TSP/` must be propagated to `article/`.
3. `IEEE_TSP/main.tex` must keep section content and TikZ figures inlined directly in the file. External binary figures and bibliography files remain allowed dependencies.
4. The writing style remains version 1.0: describe the work as the current article, avoid references to internal production phases, and use positive formulations unless a technical contrast requires otherwise.
5. After synchronization, compile both manuscript surfaces and check that labels, citations, tables, and figures still resolve.

Operational note: when regenerating `IEEE_TSP/main.tex`, use the current `article/sections/*.tex` content as the scientific source and inline all `article/figures/*.tikz` content directly.
