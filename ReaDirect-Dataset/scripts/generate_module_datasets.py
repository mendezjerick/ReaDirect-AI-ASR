#!/usr/bin/env python3
"""Generate expanded ReaDirect module activity CSV banks."""

from __future__ import annotations

import csv
from itertools import cycle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "modules"

M1_HEADER = [
    "id",
    "module_key",
    "activity_type",
    "sequence",
    "prompt_text",
    "expected_answer",
    "accepted_answers",
    "difficulty",
    "points",
    "is_mastery_item",
    "is_active",
]

M2_HEADER = [
    "id",
    "module_key",
    "activity_type",
    "sequence",
    "prompt_text",
    "target_word",
    "expected_answer",
    "accepted_answers",
    "word_family",
    "difficulty",
    "points",
    "is_mastery_item",
    "is_active",
]

M3_HEADER = [
    "id",
    "module_key",
    "activity_type",
    "sequence",
    "prompt_text",
    "expected_answer",
    "accepted_answers",
    "difficulty",
    "points",
    "is_mastery_item",
    "is_active",
]


LETTER_DATA = [
    ("A", "short a", "A|a|ah|short a", ["apple", "ant", "ax", "alligator"]),
    ("B", "b", "B|b|buh", ["bag", "bat", "bed", "bell"]),
    ("C", "hard c", "C|c|kuh|hard c", ["cat", "cup", "cap", "cot"]),
    ("D", "d", "D|d|duh", ["dog", "dig", "desk", "duck"]),
    ("E", "short e", "E|e|eh|short e", ["egg", "elf", "end", "engine"]),
    ("F", "f", "F|f|fff", ["fan", "fish", "fog", "foot"]),
    ("G", "hard g", "G|g|guh|hard g", ["goat", "gum", "gate", "gift"]),
    ("H", "h", "H|h|huh", ["hat", "hen", "hill", "hop"]),
    ("I", "short i", "I|i|ih|short i", ["igloo", "ink", "insect", "inch"]),
    ("J", "j", "J|j|juh", ["jam", "jet", "jog", "jug"]),
    ("K", "k", "K|k|kuh", ["kite", "kid", "kit", "king"]),
    ("L", "l", "L|l|luh", ["lid", "lamp", "leg", "leaf"]),
    ("M", "m", "M|m|mmm", ["map", "moon", "mat", "milk"]),
    ("N", "n", "N|n|nnn", ["net", "nest", "nap", "nose"]),
    ("O", "short o", "O|o|ah|short o", ["octopus", "ox", "odd", "ostrich"]),
    ("P", "p", "P|p|puh", ["pin", "pan", "pig", "pot"]),
    ("Q", "q", "Q|q|kw", ["queen", "quilt", "quick", "quiz"]),
    ("R", "r", "R|r|ruh", ["rat", "red", "rock", "rain"]),
    ("S", "s", "S|s|sss", ["sun", "sit", "sock", "sand"]),
    ("T", "t", "T|t|tuh", ["tap", "top", "ten", "tub"]),
    ("U", "short u", "U|u|uh|short u", ["up", "under", "uncle", "umbrella"]),
    ("V", "v", "V|v|vvv", ["van", "vet", "vest", "vine"]),
    ("W", "w", "W|w|wuh", ["web", "win", "water", "wind"]),
    ("X", "x", "X|x|ks", ["x-ray", "box", "fox", "six"]),
    ("Y", "y", "Y|y|yuh", ["yak", "yam", "yes", "yellow"]),
    ("Z", "z", "Z|z|zzz", ["zip", "zoo", "zero", "zebra"]),
]

WORD_FAMILIES = {
    "at": ["cat", "bat", "hat", "mat", "rat", "sat", "pat", "flat"],
    "an": ["can", "fan", "man", "pan", "ran", "van", "tan", "plan"],
    "ap": ["cap", "map", "nap", "tap", "lap", "gap", "snap", "clap"],
    "am": ["ham", "jam", "ram", "yam", "dam", "clam", "slam", "gram"],
    "ag": ["bag", "tag", "wag", "rag", "flag", "drag", "snag", "brag"],
    "ad": ["dad", "sad", "mad", "pad", "had", "glad", "clad", "bad"],
    "ed": ["bed", "red", "fed", "led", "wed", "shed", "sled", "sped"],
    "en": ["hen", "pen", "ten", "men", "den", "then", "when", "wren"],
    "et": ["jet", "pet", "net", "wet", "set", "get", "let", "met"],
    "in": ["pin", "bin", "fin", "win", "tin", "chin", "thin", "spin"],
    "ip": ["lip", "sip", "dip", "rip", "tip", "ship", "chip", "flip"],
    "it": ["sit", "fit", "hit", "kit", "pit", "bit", "knit", "split"],
    "ig": ["big", "dig", "pig", "wig", "fig", "twig", "jig", "rig"],
    "op": ["hop", "mop", "top", "pop", "cop", "shop", "stop", "drop"],
    "ot": ["hot", "pot", "dot", "got", "cot", "spot", "plot", "slot"],
    "og": ["dog", "log", "fog", "jog", "hog", "frog", "clog", "smog"],
    "un": ["sun", "run", "fun", "bun", "nun", "spun", "stun", "pun"],
    "ug": ["bug", "rug", "hug", "mug", "jug", "plug", "slug", "snug"],
    "ub": ["cub", "tub", "rub", "sub", "club", "stub", "scrub", "grub"],
    "ill": ["hill", "fill", "will", "bill", "mill", "still", "spill", "drill"],
    "ell": ["bell", "fell", "tell", "well", "shell", "smell", "dwell", "spell"],
    "ack": ["back", "sack", "pack", "rack", "jack", "black", "track", "stack"],
    "ock": ["rock", "sock", "lock", "dock", "clock", "block", "flock", "shock"],
    "ish": ["fish", "dish", "wish", "swish", "finish", "relish", "pinkish", "smallish"],
    "ash": ["cash", "rash", "dash", "mash", "splash", "crash", "flash", "trash"],
    "all": ["ball", "call", "fall", "wall", "tall", "small", "stall", "hall"],
    "ail": ["mail", "sail", "tail", "rail", "pail", "snail", "trail", "nail"],
    "ain": ["rain", "pain", "train", "chain", "brain", "plain", "grain", "stain"],
    "ake": ["cake", "lake", "make", "bake", "rake", "shake", "snake", "flake"],
    "ice": ["nice", "rice", "mice", "slice", "price", "twice", "spice", "dice"],
    "oat": ["boat", "coat", "goat", "float", "moat", "throat", "oat", "gloat"],
    "eep": ["keep", "deep", "sleep", "sheep", "peep", "weep", "creep", "sweep"],
}

MINIMAL_PAIRS = [
    ("cat", "cot"),
    ("bat", "bag"),
    ("pin", "pan"),
    ("bit", "bat"),
    ("cap", "cup"),
    ("hop", "hip"),
    ("ten", "tan"),
    ("ship", "shop"),
    ("fan", "fin"),
    ("map", "mop"),
    ("bed", "bad"),
    ("red", "rid"),
    ("fish", "fist"),
    ("sun", "son"),
    ("rug", "rag"),
    ("pig", "big"),
    ("log", "leg"),
    ("pen", "pin"),
    ("wet", "wit"),
    ("tap", "top"),
    ("sit", "sat"),
    ("duck", "dock"),
    ("bell", "ball"),
    ("hill", "hall"),
    ("coat", "cot"),
    ("rain", "ran"),
    ("cake", "cage"),
    ("rice", "race"),
    ("sleep", "slip"),
    ("snail", "nail"),
    ("clap", "clip"),
    ("drip", "drop"),
    ("flag", "flog"),
    ("train", "tray"),
    ("brush", "blush"),
    ("chair", "share"),
    ("thin", "then"),
    ("chip", "ship"),
    ("rock", "lock"),
    ("pack", "back"),
    ("mug", "mud"),
    ("van", "fan"),
    ("jet", "get"),
    ("gate", "date"),
    ("kite", "kit"),
    ("moon", "man"),
    ("leaf", "loaf"),
    ("seed", "said"),
    ("corn", "card"),
    ("park", "bark"),
]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def accepted_sentence(sentence: str) -> str:
    no_period = sentence[:-1] if sentence.endswith(".") else sentence
    no_comma = sentence.replace(",", "")
    no_comma_no_period = no_comma[:-1] if no_comma.endswith(".") else no_comma
    choices = [sentence, no_period, no_comma, no_comma_no_period]
    unique = []
    for choice in choices:
        if choice and choice not in unique:
            unique.append(choice)
    return "|".join(unique)


def flat_words() -> list[tuple[str, str]]:
    words: list[tuple[str, str]] = []
    for family, family_words in WORD_FAMILIES.items():
        for word in family_words:
            words.append((word, family))
    return words


def generate_module1() -> list[list[object]]:
    rows: list[list[object]] = []
    letter_cycle = list(LETTER_DATA)
    word_items = [
        (letter, sound, accepted, word)
        for letter, sound, accepted, words in LETTER_DATA
        for word in words
    ]

    for index in range(100):
        letter, sound, accepted, words = letter_cycle[index % len(letter_cycle)]
        word = words[index // len(letter_cycle) % len(words)]
        rows.append([
            f"M1-HR{index + 1:03d}",
            "module_1",
            "hear_and_repeat",
            index + 1,
            f"Listen and say the {sound} sound for {letter}, as in {word}.",
            letter,
            accepted,
            "easy",
            1,
            0,
            1,
        ])

    for index in range(100):
        letter, sound, accepted, words = letter_cycle[(index * 3) % len(letter_cycle)]
        word = words[index // len(letter_cycle) % len(words)]
        rows.append([
            f"M1-SL{index + 1:03d}",
            "module_1",
            "see_letter_say_sound",
            101 + index,
            f"Look at {letter}. Say its sound, like the first sound in {word}.",
            letter,
            accepted,
            "easy",
            1,
            0,
            1,
        ])

    for index, (letter, sound, accepted, word) in zip(range(100), cycle(word_items)):
        rows.append([
            f"M1-ML{index + 1:03d}",
            "module_1",
            "match_sound_to_letter",
            201 + index,
            f"Which letter makes the first sound in {word}?",
            letter,
            accepted.split("|")[0] + "|" + accepted.split("|")[1],
            "easy",
            1,
            0,
            1,
        ])

    for index, (letter, sound, accepted, word) in zip(range(100), cycle(word_items[13:] + word_items[:13])):
        rows.append([
            f"M1-SD{index + 1:03d}",
            "module_1",
            "sound_drill",
            301 + index,
            f"Say the first sound in {word}.",
            letter,
            accepted,
            "easy",
            1,
            0,
            1,
        ])

    for index, (letter, sound, accepted, words) in zip(range(100), cycle(letter_cycle)):
        word = words[(index // len(letter_cycle)) % len(words)]
        rows.append([
            f"M1-MC{index + 1:03d}",
            "module_1",
            "mastery_check",
            401 + index,
            f"Say the sound for {letter}. Think of {word}.",
            letter,
            accepted,
            "easy",
            1,
            1,
            1,
        ])

    return rows


def generate_module2() -> list[list[object]]:
    rows: list[list[object]] = []
    words = flat_words()

    for index, (word, family) in zip(range(100), cycle(words)):
        rows.append([
            f"M2-RW{index + 1:03d}",
            "module_2",
            "read_word",
            index + 1,
            f"Read the word {word}.",
            word,
            word,
            word,
            family,
            "easy",
            1,
            0,
            1,
        ])

    family_items = [
        (family, word)
        for family, family_words in WORD_FAMILIES.items()
        for word in family_words
    ]
    for index, (family, word) in zip(range(100), cycle(family_items[31:] + family_items[:31])):
        rows.append([
            f"M2-WF{index + 1:03d}",
            "module_2",
            "word_family_drill",
            101 + index,
            f"Read a word in the {family} family: {word}.",
            word,
            word,
            word,
            family,
            "easy",
            1,
            0,
            1,
        ])

    minimal_targets = []
    for first, second in MINIMAL_PAIRS:
        minimal_targets.append((first, second))
        minimal_targets.append((second, first))
    for index, (target, contrast) in zip(range(100), cycle(minimal_targets)):
        family = target[-3:] if len(target) > 3 else target[-2:]
        rows.append([
            f"M2-MP{index + 1:03d}",
            "module_2",
            "minimal_pair",
            201 + index,
            f"Read this word carefully: {target}. It is different from {contrast}.",
            target,
            target,
            target,
            family,
            "easy",
            1,
            0,
            1,
        ])

    for index, (word, family) in zip(range(100), cycle(words[77:] + words[:77])):
        rows.append([
            f"M2-WA{index + 1:03d}",
            "module_2",
            "word_accuracy_challenge",
            301 + index,
            f"Read the word smoothly and clearly: {word}.",
            word,
            word,
            word,
            family,
            "easy",
            1,
            0,
            1,
        ])

    for index, (word, family) in zip(range(100), cycle(words[139:] + words[:139])):
        rows.append([
            f"M2-MC{index + 1:03d}",
            "module_2",
            "mastery_check",
            401 + index,
            f"Read the word {word}.",
            word,
            word,
            word,
            family,
            "easy",
            1,
            1,
            1,
        ])

    return rows


def simple_sentence_pool() -> list[str]:
    subjects = ["Ana", "Ben", "Mia", "Sam", "Leo", "Nina", "Omar", "Lila", "Mateo", "Asha"]
    actions = [
        "reads a small book",
        "packs a blue bag",
        "feeds the brown dog",
        "draws a bright sun",
        "holds a red cup",
        "finds a green leaf",
        "opens the class door",
        "shares a soft mat",
    ]
    return [f"{subject} {action}." for subject in subjects for action in actions]


def coach_sentence_pool() -> list[str]:
    starters = ["The cat", "The dog", "The bird", "The fish", "The class", "The team", "My friend", "Our group"]
    endings = [
        "runs home after lunch",
        "sits near the window",
        "looks at the red ball",
        "waits by the school gate",
        "helps clean the room",
        "reads under the tree",
        "carries water to the garden",
        "smiles at the new picture",
        "walks slowly to the bus",
        "plays a quiet game",
    ]
    return [f"{starter} {ending}." for starter in starters for ending in endings]


def timed_sentence_pool() -> list[str]:
    subjects = ["I", "We", "They", "Ana", "Ben", "Mia", "Sam", "Leo", "Nina", "Omar"]
    actions = [
        "can read this word",
        "will walk to class",
        "can run to the gate",
        "will pack the bag",
        "can sit on the mat",
        "will help at home",
        "can find the book",
        "will feed the fish",
    ]
    return [f"{subject} {action}." for subject in subjects for action in actions]


def pause_sentence_pool() -> list[str]:
    first_parts = ["Sam runs", "Mia reads", "Ben looks", "Ana waits", "Leo jumps", "Nina smiles", "Omar waves", "Lila listens"]
    second_parts = [
        "then sits",
        "then draws",
        "then points",
        "then claps",
        "then rests",
        "then answers",
        "then helps",
        "then walks",
        "then shares",
        "then writes",
    ]
    return [f"{first}, {second}." for first in first_parts for second in second_parts]


def fluency_sentence_pool() -> list[str]:
    starts = ["Ana and Ben", "Mia and Sam", "Leo and Nina", "Omar and Lila", "Mateo and Asha"]
    middles = [
        "read the map",
        "carry the basket",
        "water the garden",
        "clean the shelf",
        "pack the books",
        "watch the clouds",
        "count the shells",
        "draw the banner",
    ]
    endings = ["before lunch", "after class", "near the gate", "under the tree", "with calm voices"]
    return [f"{start} {middle} {ending}." for start in starts for middle in middles for ending in endings]


def generate_module3() -> list[list[object]]:
    rows: list[list[object]] = []

    segments = [
        ("M3-RS", "read_sentence", 1, 80, simple_sentence_pool(), lambda sentence: sentence),
        ("M3-RC", "read_with_coach", 81, 80, coach_sentence_pool(), lambda sentence: f"Read with me: {sentence}"),
        ("M3-TS", "timed_sentence_reading", 161, 80, timed_sentence_pool(), lambda sentence: f"Read smoothly: {sentence}"),
        ("M3-PP", "pause_practice", 241, 80, pause_sentence_pool(), lambda sentence: f"Pause after the comma: {sentence}"),
        ("M3-FC", "fluency_challenge", 321, 80, fluency_sentence_pool(), lambda sentence: f"Read with a smooth voice: {sentence}"),
    ]

    mastery_source = (
        simple_sentence_pool()[:25]
        + coach_sentence_pool()[:25]
        + timed_sentence_pool()[:20]
        + pause_sentence_pool()[:15]
        + fluency_sentence_pool()[:15]
    )

    for prefix, activity_type, start, count, sentences, prompt_builder in segments:
        for index, sentence in zip(range(count), cycle(sentences)):
            rows.append([
                f"{prefix}{index + 1:03d}",
                "module_3",
                activity_type,
                start + index,
                prompt_builder(sentence),
                sentence,
                accepted_sentence(sentence),
                "easy",
                1,
                0,
                1,
            ])

    for index, sentence in zip(range(100), cycle(mastery_source)):
        rows.append([
            f"M3-MC{index + 1:03d}",
            "module_3",
            "mastery_check",
            401 + index,
            sentence,
            sentence,
            accepted_sentence(sentence),
            "easy",
            1,
            1,
            1,
        ])

    return rows


def main() -> None:
    write_csv(MODULES / "module1_letter_sound_activities.csv", M1_HEADER, generate_module1())
    write_csv(MODULES / "module2_word_reading_activities.csv", M2_HEADER, generate_module2())
    write_csv(MODULES / "module3_sentence_fluency_activities.csv", M3_HEADER, generate_module3())
    print("Generated 500 rows for each module activity CSV.")


if __name__ == "__main__":
    main()
