"""Every non-name phrase of a task, as slots of exact paraphrases; each slot holds some templates out of training.

``SLOTS[slot] = (training templates, held-out templates)``. Training and i.i.d. evaluation data draw only
training templates; the held-out-wording split draws only held-out ones. Every template states its
estimand explicitly (dependence, or the effect of an intervention), so paraphrases never change an answer.
Placeholders: ``{x}``, ``{y}``, ``{z}`` variable names or sets; ``{proc}`` an adjustment procedure;
``{where}`` a conditioning phrase; ``{s}`` a structural statement; ``{a}``, ``{b}`` variable names.
"""

from __future__ import annotations

SLOTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # --- graph descriptions ---
    "header": (
        (
            "A closed system has {n} variables: {names}. Its causal structure is:",
            "Consider {n} variables, {names}, with the following causal relationships:",
        ),
        ("{n} variables are studied: {names}. The direct causal links among them are:",),
    ),
    "closing": (
        (
            "There are no other causal relationships and no hidden common causes.",
            "These are all of the causal relationships, and no unmeasured variable affects two or more of them.",
        ),
        ("No other causal links exist, and there are no unobserved confounders.",),
    ),
    "faithful": (
        (
            "Every causal relationship is real, with no exact cancellations.",
            "Each causal link has a real effect, and effects along different paths never cancel exactly.",
        ),
        ("No dependence ever vanishes through an exact cancellation of effects.",),
    ),
    # --- independence ---
    "where_given": (
        (
            "among units that share the same values of {z}",
            "after conditioning on {z}",
            "within groups defined by {z}",
        ),
        ("once the values of {z} are known", "within each stratum of {z}"),
    ),
    "where_none": (
        ("in the population as a whole", "without conditioning on anything"),
        ("marginally, with no other variable taken into account",),
    ),
    "ask_independent": (
        ("Are {x} and {y} statistically independent {where}?", "Is {x} statistically independent of {y} {where}?"),
        ("Does learning the value of {x} leave the probability distribution of {y} unchanged {where}?",),
    ),
    "ask_associated": (
        ("Would {x} and {y} be statistically associated {where}?", "Are {x} and {y} statistically dependent {where}?"),
        ("Would learning the value of {x} change the probability distribution of {y} {where}?",),
    ),
    # --- adjustment ---
    "proc_set": (
        (
            "compare units with different values of {x} within groups that share the same values of {z}, then "
            "average those comparisons over the groups, weighted by group size",
            "adjust for {z} by stratifying on {z}, comparing {x} groups within each stratum and averaging over "
            "strata using the population distribution of {z}",
            "control for {z} with the adjustment formula, summing over the values of {z} weighted by how common "
            "each is",
        ),
        (
            "standardize over {z}: estimate the contrast between {x} groups separately for every combination of "
            "values of {z}, then average those estimates using the share of the population in each combination",
        ),
    ),
    "proc_empty": (
        (
            "compare units with different values of {x} directly, adjusting for nothing",
            "compare {x} groups without adjusting for any other variable",
            "use the raw difference between units with different values of {x}, with no adjustment",
        ),
        ("take the unadjusted contrast between units with different values of {x}",),
    ),
    "ask_single": (
        (
            "Researchers want the causal effect of {x} on {y} from observational data. They {proc}. With unlimited "
            "data, would this recover the causal effect exactly?",
            "An analyst wants to learn how {y} would change if {x} were changed by intervention, using only "
            "observational data. The analyst plans to {proc}. Given unlimited data, does this give the causal "
            "effect of {x} on {y} without bias?",
            "Goal: the total causal effect of {x} on {y}, estimated from observational data. Approach: {proc}. "
            "With unlimited data, is the approach unbiased?",
        ),
        (
            "Using observational data only, a team will {proc}. Given unlimited data, will the result equal the "
            "effect that intervening on {x} has on {y}?",
        ),
    ),
    "short_set": (
        ("adjusts for {z}", "stratifies on {z} and averages over the distribution of {z}", "controls for {z}"),
        ("standardizes over {z}",),
    ),
    "short_empty": (
        ("adjusts for nothing", "uses the unadjusted comparison", "does not adjust for any variable"),
        ("reports the raw contrast",),
    ),
    "ask_compare": (
        (
            "Two analysts estimate the causal effect of {x} on {y} from observational data. Analyst 1 {proc1}; "
            "analyst 2 {proc2}. With unlimited data, whose estimate is unbiased: 1, 2, Both or Neither?",
            "Analyst 1 and analyst 2 both want the effect on {y} of intervening on {x}, using observational data. "
            "Analyst 1 {proc1}, while analyst 2 {proc2}. Assuming unlimited data, which of them recovers the causal "
            "effect: 1, 2, Both or Neither?",
        ),
        (
            "Each aiming at the causal effect of {x} on {y} from observational data, analyst 1 {proc1}; analyst 2 "
            "{proc2}. With unlimited data, whose result equals that effect: 1, 2, Both or Neither?",
        ),
    ),
    # --- effect ---
    "mechanisms": (
        (
            "Every variable is binary (1 = present, 0 = absent). The causal mechanisms are:",
            "All variables are binary (1 means present, 0 absent), and they are generated as follows:",
        ),
        ("Each variable takes the value 1 (present) or 0 (absent). The mechanisms are:",),
    ),
    "cpt_row": (
        ("P({v}=1 | {given}) = {p}", "{v}=1 with probability {p} when {given}"),
        ("among units with {given}, a share {p} have {v}=1",),
    ),
    "cpt_root": (
        ("P({v}=1) = {p}", "{v}=1 with probability {p}"),
        ("a share {p} of all units have {v}=1",),
    ),
    "ask_causal": (
        (
            "If {x} were set to 1 for everyone, rather than to 0 for everyone, by how much would the probability "
            "that {y}=1 change? Give P({y}=1 | do({x}=1)) - P({y}=1 | do({x}=0)).",
            "Compute the average causal effect of {x} on {y}: P({y}=1 | do({x}=1)) - P({y}=1 | do({x}=0)).",
        ),
        (
            "A policy sets {x} to 1 for the whole population; the alternative policy sets it to 0. How much higher "
            "is the probability that {y}=1 under the first policy than under the second? (A negative number means "
            "lower.)",
        ),
    ),
    "ask_observed": (
        (
            "What is P({y}=1 | {x}=1) - P({y}=1 | {x}=0) in this population?",
            "Compute the observed difference P({y}=1 | {x}=1) - P({y}=1 | {x}=0), with no intervention.",
        ),
        (
            "In passively observed data, how much more common is {y}=1 among units with {x}=1 than among units "
            "with {x}=0? (A negative number means less common.)",
        ),
    ),
    "ask_sign": (
        (
            "Would setting {x} to 1 for everyone, rather than to 0, make {y}=1 more likely?",
            "Is P({y}=1 | do({x}=1)) greater than P({y}=1 | do({x}=0))?",
        ),
        ("If a policy forced {x}=1 for all units instead of {x}=0, would {y}=1 become more common?",),
    ),
    "ask_gap": (
        (
            "Is the observed difference P({y}=1 | {x}=1) - P({y}=1 | {x}=0) larger, as a signed number, than the "
            "causal effect P({y}=1 | do({x}=1)) - P({y}=1 | do({x}=0))?",
            "As signed numbers, does P({y}=1 | {x}=1) - P({y}=1 | {x}=0) exceed "
            "P({y}=1 | do({x}=1)) - P({y}=1 | do({x}=0))?",
        ),
        (
            "Compare the observational contrast P({y}=1 | {x}=1) - P({y}=1 | {x}=0) with the interventional "
            "contrast P({y}=1 | do({x}=1)) - P({y}=1 | do({x}=0)). Is the observational one larger, as a signed "
            "number?",
        ),
    ),
    # --- discovery ---
    "study": (
        (
            "A study measured {n} variables: {names}. They form a closed causal system with no hidden common causes, "
            "and every dependence among them reflects the causal structure (no exact cancellations).",
            "{n} variables were measured: {names}. No unmeasured variable affects two or more of them, and no "
            "dependence vanishes through an exact cancellation of effects.",
        ),
        (
            "Researchers recorded {n} variables, {names}, which make up a causal system without unobserved "
            "confounders; all of their dependences are faithful to the causal structure.",
        ),
    ),
    "independences": (
        ("The independences that hold are exactly these:", "These are all the independences among them:"),
        ("Only the following independences hold:",),
    ),
    "indep_given": (
        ("{a} and {b} are independent given {z}.", "{a} is independent of {b} given {z}."),
        ("Once {z} is known, {a} and {b} are independent.",),
    ),
    "indep_none": (
        ("{a} and {b} are independent.", "{a} is independent of {b}."),
        ("{a} and {b} are unconditionally independent.",),
    ),
    "dependent_rest": (
        (
            "Every other pair is dependent given any other set of variables.",
            "Any pair not listed is dependent no matter which other variables are conditioned on.",
        ),
        ("All remaining pairs stay dependent under every conditioning set.",),
    ),
    "no_independences": (
        (
            "No independences hold: every pair of variables is dependent given any set of the others.",
            "There are no independences: every pair stays dependent whatever else is conditioned on.",
        ),
        ("Not a single independence holds among them, under any conditioning set.",),
    ),
    "ask_must": (
        ("Must it be true that {s}?", "Is it necessarily true that {s}?"),
        ("Do these facts guarantee that {s}?",),
    ),
    "ask_could": (
        ("Is it possible that {s}?", "Could it be true that {s}?"),
        ("Is it consistent with these facts that {s}?",),
    ),
    "direct": (
        ("{a} directly causes {b}", "{a} is a direct cause of {b}"),
        ("there is a direct causal link from {a} to {b}",),
    ),
    "cause": (
        ("{a} causes {b}, directly or through other variables", "{a} is a cause, direct or indirect, of {b}"),
        ("{b} is causally downstream of {a}, directly or indirectly",),
    ),
    "common_cause": (
        (
            "some other variable in the system causes both {a} and {b}, directly or indirectly",
            "{a} and {b} share a common cause, direct or indirect, among the other variables",
        ),
        ("a third measured variable is a cause, direct or indirect, of both {a} and {b}",),
    ),
    "common_effect": (
        (
            "{a} and {b} both directly cause some third variable",
            "some third variable has both {a} and {b} as direct causes",
        ),
        ("{a} and {b} have a common direct effect",),
    ),
}

GRAPH_STYLES: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]] = (
    (
        ("edge", "{P} directly causes {c}."),
        ("edge", "{C} is directly affected by {p}."),
        ("parents", "{C} depends directly on {ps}."),
        ("arrows", "{p} -> {c}"),
    ),
    (("children", "{P} directly affects {cs}."), ("edge", "There is a direct causal arrow from {p} to {c}.")),
)
"""(training, held-out) graph description styles: one sentence per edge, per child, per parent, or arrow lines."""

BANNED = ("case by case", "method 1", "method 2", "more correct", "self-contained", "hypothetical world")
"""Phrases of the far-transfer benchmark (CLadder) that no template may contain."""
