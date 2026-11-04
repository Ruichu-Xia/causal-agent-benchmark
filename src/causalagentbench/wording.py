"""Every non-name phrase of a data-analysis task, as slots of exact paraphrases with some held out of training.

``SLOTS[slot] = (training templates, held-out templates)``: training and
i.i.d. evaluation data draw training templates, and the held-out-wording split draws held-out ones. No
template changes what a question asks. Placeholders: ``{rows}``, ``{n}``, ``{columns}``; ``{x}``, ``{y}``,
``{z}`` column names or sets; ``{where}`` a conditioning phrase; ``{proc}``, ``{proc1}``, ``{proc2}`` adjustment
procedures; ``{options}`` lettered options.
"""

from __future__ import annotations

SLOTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # --- shared context ---
    "intro": (
        (
            "A dataset of {rows} observations of {n} variables is loaded in the python tool as the pandas "
            "DataFrame `df`. Its columns are: {columns}.",
            "The python tool has a pandas DataFrame `df` with {rows} rows, one column per variable: {columns}.",
        ),
        ("You are given {rows} observations of {n} variables ({columns}), available in the python tool as `df`.",),
    ),
    "assumptions": (
        (
            "The data come from a linear causal system with independent Gaussian noise. There are no feedback "
            "loops and no unmeasured common causes, effects along different paths never cancel exactly, and "
            "every variable has variance one in the population.",
            "Assume a linear, acyclic causal model with independent Gaussian noise, no hidden confounders and no "
            "exact cancellation of effects. Each variable has population variance one.",
        ),
        (
            "The generating process is a linear structural causal model without cycles, with Gaussian noise, "
            "with every common cause measured, and with no effects that cancel exactly; all variables have unit "
            "population variance.",
        ),
    ),
    "names": (
        (
            "Each column is named after a real-world concept. Every direct causal link in this system is one "
            "that the concepts make plausible, in that direction, but some plausible links are absent. Effect "
            "sizes and signs are synthetic.",
            "The column names are real-world concepts. Each direct cause in this system is plausible from the "
            "names, in that direction; not every plausible link is present, and the sizes and signs of the "
            "effects are simulated.",
        ),
        (
            "Variables are named after real concepts: whenever one directly causes another here, that is "
            "plausible from what they mean, in that direction, though some plausible links do not exist. The "
            "strengths and signs of effects are made up.",
        ),
    ),
    "tools": (
        (
            "Analyze the data with the python tool. Besides numpy (np), pandas (pd), scipy and statsmodels, it "
            "provides partial_corr(df, x, y, given=()), pc(df, alpha=0.01) and ols(df, y, X); help() on any of "
            "them describes it.",
            "Use the python tool to investigate. It has numpy as np, pandas as pd, scipy, statsmodels, and the "
            "helpers partial_corr(df, x, y, given=()), pc(df, alpha=0.01) and ols(df, y, X) (see help() on each).",
        ),
        (
            "Work in the python tool, where np, pd, scipy, statsmodels and three helpers are available: "
            "partial_corr(df, x, y, given=()), pc(df, alpha=0.01) and ols(df, y, X); call help() on a helper for "
            "its documentation.",
        ),
    ),
    # --- independence ---
    "where_given": (
        ("given {z}", "after conditioning on {z}"),
        ("once {z} are held fixed",),
    ),
    "where_none": (
        ("without conditioning on anything", "marginally"),
        ("in the population as a whole",),
    ),
    "ask_independent": (
        (
            "In the population, are {x} and {y} independent {where}?",
            "Is {x} statistically independent of {y} {where}, in the population?",
        ),
        ("In the population, does {x} carry no information about {y} {where}?",),
    ),
    "ask_associated": (
        (
            "In the population, are {x} and {y} dependent {where}?",
            "Is {x} statistically associated with {y} {where}, in the population?",
        ),
        ("In the population, does {x} carry information about {y} {where}?",),
    ),
    # --- structure ---
    "ask_structure": (
        (
            "Which of these is true in this system?\n{options}",
            "Which statement about this system is correct?\n{options}",
        ),
        ("Exactly one of the following holds in this system. Which?\n{options}",),
    ),
    "causes": (
        ("{x} causes {y}, directly or through other variables", "{x} is a cause of {y}, directly or indirectly"),
        ("changing {x} would change {y}, directly or via other variables",),
    ),
    "neither": (
        ("neither causes the other", "neither of {x} and {y} causes the other"),
        ("changing either one would leave the other unchanged",),
    ),
    # --- adjustment ---
    "proc_set": (
        (
            "regress {y} on {x} and {z} and take the coefficient of {x}",
            "estimate the effect of {x} on {y} by linear regression of {y} on {x}, adjusting for {z}",
        ),
        ("fit a linear model of {y} on {x} with {z} as covariates and read off the coefficient of {x}",),
    ),
    "proc_empty": (
        ("regress {y} on {x} alone and take the coefficient of {x}", "use the simple regression slope of {y} on {x}"),
        ("fit a linear model of {y} on {x} with no covariates and read off the coefficient of {x}",),
    ),
    "ask_single": (
        (
            "To estimate the total causal effect of {x} on {y}, an analyst will {proc}. With unlimited data, "
            "would this recover the causal effect?",
            "An analyst wants the total causal effect of {x} on {y}, and will {proc}. Is this estimate unbiased "
            "for the causal effect, given unlimited data?",
        ),
        (
            "Someone plans to {proc} in order to learn the total causal effect of {x} on {y}. Would that be "
            "consistent for the causal effect as the sample grows without bound?",
        ),
    ),
    "ask_compare": (
        (
            "Two analysts estimate the total causal effect of {x} on {y}. Analyst 1 will {proc1}; analyst 2 will "
            "{proc2}. With unlimited data, whose estimate recovers the causal effect: 1, 2, Both or Neither?",
        ),
        (
            "To learn the total causal effect of {x} on {y}, analyst 1 will {proc1}, and analyst 2 will {proc2}. "
            "Which of them would be consistent for the causal effect: 1, 2, Both or Neither?",
        ),
    ),
    # --- effect ---
    "ask_effect": (
        (
            "If {x} were set one standard deviation higher by an intervention, by how many standard deviations "
            "would {y} change on average? Give the total effect, including through other variables.",
            "What is the total causal effect of {x} on {y}: the average change in {y}, in standard deviations, "
            "when {x} is raised by one standard deviation through an intervention?",
        ),
        (
            "Suppose an intervention increases {x} by one standard deviation. What is the expected change in "
            "{y}, in standard deviations, counting every causal path?",
        ),
    ),
}

FORMATS = {
    "yesno": "Yes or No",
    "letter": "A, B or C",
    "choice": "1, 2, Both or Neither",
    "number": "a number rounded to two decimal places",
}


def instruction(kind: str) -> str:
    """The final-line instruction for an answer format."""
    return f'When you are done, end your reply with a final line "Answer: <answer>", where <answer> is {FORMATS[kind]}.'
