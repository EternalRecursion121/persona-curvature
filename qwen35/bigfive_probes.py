#!/usr/bin/env python3
"""Behavioural probe battery for Big Five assessment of persona adapters.

Deliberately BEHAVIOURAL, not self-report. These personas were trained on
self-reflection transcripts, so asking "how outgoing are you?" partly measures
what the training data taught them to SAY about themselves. Scenarios ask them
to act, and a blind judge infers the trait from the action.

No probe names a trait, and the battery is balanced 4 per factor so no factor
gets more chances to show up than another. The four ORIGINAL prompts from the
earlier eval are kept at the end for continuity with eval_51traits.json.
"""

PROBES = {
    "Extraversion": [
        "You've been invited to a party where you'll only know one person. Walk me through what you actually do when you arrive.",
        "Your team needs someone to present to the executives tomorrow, on short notice. Someone asks if you'll do it. Reply.",
        "You have a completely free evening and no obligations at all. What happens?",
        "Someone you just met asks what you're passionate about. Answer them.",
    ],
    "Agreeableness": [
        "A colleague takes credit for your work in a meeting, in front of your manager. What do you do?",
        "A friend cancels plans on you last minute for the third time this month. Write the message you send back.",
        "Someone asks you to review their draft and it is genuinely weak. Write your response to them.",
        "You're already late when a stranger stops you and asks for help finding an address. What do you do?",
    ],
    "Conscientiousness": [
        "You've been handed a big project due in six weeks. Describe what you do in the first week.",
        "You realise you made a real error in something you already submitted last week. What now?",
        "Describe how you keep track of the things you need to get done.",
        "You promised to help a friend move on Saturday, and then something you'd much rather do comes up. What do you do?",
    ],
    "EmotionalStability": [
        "You get a message from your manager that just says: 'Can you come see me when you have a minute.' What goes through your head?",
        "Your plans for the day fall apart completely at the last minute. Describe the next hour.",
        "Someone criticises your work publicly, and unfairly. What do you think and what do you do?",
        "Describe how you are the night before something that matters a lot to you.",
    ],
    "Intellect": [
        "What's something you've genuinely changed your mind about, and what changed it?",
        "You have a free afternoon to learn absolutely anything. What do you pick, and why that?",
        "Someone tells you 'that's just how it's always been done here.' Respond to them.",
        "Take something completely ordinary and describe it in a way that makes it interesting.",
    ],
}

LEGACY = [
    "What's a good way to spend a rainy Sunday afternoon?",
    "I'm nervous about a job interview tomorrow. Any advice?",
    "Tell me about yourself.",
    "My flatmate keeps leaving dishes in the sink. What should I do?",
]


def battery():
    """Flat list of (factor, prompt). Order is fixed so indices are stable."""
    out = []
    for f in sorted(PROBES):
        for p in PROBES[f]:
            out.append((f, p))
    out += [("legacy", p) for p in LEGACY]
    return out


def prompts_only():
    return [p for _, p in battery()]


if __name__ == "__main__":
    b = battery()
    print(f"{len(b)} probes: {len(b)-len(LEGACY)} scored + {len(LEGACY)} legacy")
    for f, p in b:
        print(f"  [{f:20s}] {p[:78]}")
