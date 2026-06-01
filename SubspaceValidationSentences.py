"""
SubspaceValidationSentences.py
======================
Sentence sets for subspace validation.
Includes sanity-check sentences and per-myth-type sentences.

"""

# ── Sanity check sentences ─────────────────────────────────────────────────────
KNOWN_MYTH = [
    "She was wearing a short skirt so she was asking for it.",
    "If she was drunk she should have known better.",
    "He couldn't help himself, he was intoxicated.",
    "She didn't fight back so she must have wanted it.",
    "Women who dress provocatively are inviting trouble.",
    "A lot of rape accusations are just regret the morning after.",
    "If she really didn't want it she would have screamed.",
    "Men have strong urges they can't always control.",
]

KNOWN_DEBUNKED = [
    "Clothing is never an invitation for sexual assault.",
    "Intoxication does not equal consent.",
    "The perpetrator is solely responsible for the assault.",
    "Lack of resistance does not imply consent.",
    "Sexual violence is never the victim's fault.",
    "False reports of sexual assault are extremely rare.",
    "Freezing is a common trauma response and does not mean consent.",
    "Sexual assault is about power and control, not uncontrollable desire.",
]

NEUTRAL = [
    "The weather was cold yesterday.",
    "She ordered a coffee at the café.",
    "The meeting was scheduled for Tuesday afternoon.",
    "He took the bus to work.",
    "The report was submitted on time.",
]

# ── Per-myth-type sentences ────────────────────────────────────────────────────
# Each entry: first two sentences are myth-reinforcing, last two are myth-rejecting.
MYTH_TYPE_SENTENCES = {
    "clothing": [
        "She was wearing a revealing outfit so she invited trouble.",
        "Her short skirt was basically asking for it.",
        "Clothing is never an invitation for sexual assault.",
        "What a person wears does not imply consent.",
    ],
    "victim_intoxication": [
        "She was drunk so she's partly responsible.",
        "If she hadn't been drinking this wouldn't have happened.",
        "Intoxication does not equal consent.",
        "A victim's intoxication never reduces the perpetrator's responsibility.",
    ],
    "perpetrator_intoxication": [
        "He was drunk so he couldn't help himself.",
        "The alcohol made him lose control.",
        "A perpetrator's intoxication does not excuse the assault.",
        "Being intoxicated does not remove a perpetrator's responsibility.",
    ],
    "resistance": [
        "She didn't fight back so she must have wanted it.",
        "If she really objected she would have resisted more.",
        "Lack of physical resistance does not equal consent.",
        "Freezing is a trauma response and does not imply consent.",
    ],
}
