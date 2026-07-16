"""
MythStatements.py
=====================
Myth statements for Turn-2 context injection.
Keyed by (myth_type, frame, dose) for direct lookup.

Dose 1 = single sentence.
Dose 2 = three sentences.

Statements updated to third-person to match narrative register.
"""

# Raw statements nested by dose → frame → myth_type
RAW = {
    1: {
        "NegMyth": {
            "clothing":                "The victim was not wearing a revealing outfit.",
            "victim_intoxication":     "The victim was not intoxicated.",
            "perpetrator_intoxication":"The perpetrator was not intoxicated.",
            "resistance":              "The victim resisted the perpetrator.",
        },
        "NegNonMyth": {
            "clothing":                "The victim was not wearing a modest outfit.",
            "victim_intoxication":     "The victim was not sober.",
            "perpetrator_intoxication":"The perpetrator was not sober.",
            "resistance":              "The victim yielded to the perpetrator.",
        },
        "PosMyth": {
            "clothing":                "The victim was wearing a revealing outfit.",
            "victim_intoxication":     "The victim was intoxicated.",
            "perpetrator_intoxication":"The perpetrator was intoxicated.",
            "resistance":              "The victim did not resist the perpetrator.",
        },
        "PosNonMyth": {
            "clothing":                "The victim was wearing a modest outfit.",
            "victim_intoxication":     "The victim was sober.",
            "perpetrator_intoxication":"The perpetrator was sober.",
            "resistance":              "The victim did not yield to the perpetrator.",
        },
    },
    2: {
        "NegMyth": {
            "clothing":
                "The victim was not wearing a revealing outfit. "
                "Their outfit did not expose much of their skin and body shape. "
                "Others observed that their clothing was not bold in style and "
                "did not have minimum coverage.",
            "victim_intoxication":
                "The victim was not intoxicated. "
                "They had not indulged in consuming alcohol, and their senses "
                "were not impaired. "
                "Others observed that they did not have an unsteady walk, "
                "slurred speech, and blurred vision.",
            "perpetrator_intoxication":
                "The perpetrator was not intoxicated. "
                "The perpetrator had not indulged in consuming alcohol, and "
                "their senses were not impaired. "
                "Others observed that the perpetrator did not have an unsteady "
                "walk, slurred speech, and blurred vision.",
            "resistance":
                "The victim resisted the perpetrator. "
                "They pushed them and stepped away to prevent them from taking "
                "control. "
                "They shouted for the perpetrator to stop and hindered them "
                "from continuing.",
        },
        "NegNonMyth": {
            "clothing":
                "The victim was not wearing a modest outfit. "
                "Their outfit did not cover much of their skin and body shape. "
                "Others observed that their clothing was not conservative in "
                "style and did not have maximum coverage.",
            "victim_intoxication":
                "The victim was not sober. "
                "They had not refrained from consuming alcohol, and their "
                "senses were not unaffected. "
                "Others observed that they did not have a stable walk, "
                "articulate speech, and clear vision.",
            "perpetrator_intoxication":
                "The perpetrator was not sober. "
                "The perpetrator had not refrained from consuming alcohol, and "
                "their senses were not unaffected. "
                "Others observed that the perpetrator did not have a stable "
                "walk, articulate speech, and clear vision.",
            "resistance":
                "The victim yielded to the perpetrator. "
                "They allowed them to take control and stood still. "
                "They remained silent and let the perpetrator continue "
                "unhindered.",
        },
        "PosMyth": {
            "clothing":
                "The victim was wearing a revealing outfit. "
                "Their outfit exposed much of their skin and body shape. "
                "Others observed that their clothing was bold in style and "
                "had minimum coverage.",
            "victim_intoxication":
                "The victim was intoxicated. "
                "They had indulged in consuming alcohol, and their senses "
                "were impaired. "
                "Others observed that they had an unsteady walk, slurred "
                "speech, and blurred vision.",
            "perpetrator_intoxication":
                "The perpetrator was intoxicated. "
                "The perpetrator had indulged in consuming alcohol, and their "
                "senses were impaired. "
                "Others observed that the perpetrator had an unsteady walk, "
                "slurred speech, and blurred vision.",
            "resistance":
                "The victim did not resist the perpetrator. "
                "They did not push them and stepped away to prevent them from "
                "taking control. "
                "They did not shout for the perpetrator to stop and did not "
                "hinder them from continuing.",
        },
        "PosNonMyth": {
            "clothing":
                "The victim was wearing a modest outfit. "
                "Their outfit covered much of their skin and body shape. "
                "Others observed that their clothing was conservative in style "
                "and had maximum coverage.",
            "victim_intoxication":
                "The victim was sober. "
                "They had refrained from consuming alcohol, and their senses "
                "were unaffected. "
                "Others observed that they had a stable walk, articulate "
                "speech, and clear vision.",
            "perpetrator_intoxication":
                "The perpetrator was sober. "
                "The perpetrator had refrained from consuming alcohol, and "
                "their senses were unaffected. "
                "Others observed that the perpetrator had a stable walk, "
                "articulate speech, and clear vision.",
            "resistance":
                "The victim did not yield to the perpetrator. "
                "They did not allow them to take control and did not stay "
                "still. "
                "They did not remain silent and did not let the perpetrator "
                "continue unhindered.",
        },
    },
}

# Flat lookup: (myth_type, frame, dose) → statement string
MYTH_STATEMENTS: dict[tuple, str] = {
    (myth_type, frame, dose): statement
    for dose, frames in RAW.items()
    for frame, myths in frames.items()
    for myth_type, statement in myths.items()
}