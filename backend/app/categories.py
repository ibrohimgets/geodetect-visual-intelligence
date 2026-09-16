"""
Grouping the 80 COCO classes into categories a person would actually use.

The detector thinks in terms of "class 2", the operator thinks in terms of
"vehicles". This module is the translation layer, and it is deliberately
general purpose -- everyday indoor and outdoor objects, nothing domain-specific.

Two consumers depend on it:
  * the filter chips in the UI
  * the assistant, so "show me the vehicles" resolves to a real set of classes
    rather than the model guessing which words count as vehicles
"""

from __future__ import annotations

# Ordered so the UI chips come out in a sensible order.
CATEGORIES: dict[str, list[str]] = {
    "people": ["person"],
    "vehicles": [
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    ],
    "animals": [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe",
    ],
    "furniture": [
        "chair", "couch", "bed", "dining table", "toilet", "potted plant",
    ],
    "outdoor": [
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    ],
    "personal items": [
        "backpack", "umbrella", "handbag", "tie", "suitcase", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ],
    "electronics": [
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "refrigerator", "sink",
    ],
    "kitchen": [
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    ],
    "food": [
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake",
    ],
    "sports": [
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
    ],
}

# The chips shown by default. The rest stay available to the assistant and to
# the class filter, they just do not each get a button.
PRIMARY_CATEGORIES = ["people", "vehicles", "animals", "furniture", "outdoor"]

# Human-readable labels for the UI.
CATEGORY_LABELS = {
    "people": "People",
    "vehicles": "Vehicles",
    "animals": "Animals",
    "furniture": "Furniture",
    "outdoor": "Outdoor Objects",
    "personal items": "Personal Items",
    "electronics": "Electronics",
    "kitchen": "Kitchen",
    "food": "Food",
    "sports": "Sports",
    "other": "Other",
}

# Reverse index, built once: class name -> category name.
_CLASS_TO_CATEGORY: dict[str, str] = {
    cls: category for category, classes in CATEGORIES.items() for cls in classes
}


def category_of(class_name: str) -> str:
    """Which category a COCO class belongs to. Unmapped classes land in 'other'."""
    return _CLASS_TO_CATEGORY.get(class_name, "other")


def classes_in(category: str) -> list[str]:
    """Every class in a category. Unknown category names give an empty list."""
    return list(CATEGORIES.get(category.strip().lower(), []))


def resolve_categories(names: list[str]) -> set[str]:
    """Turn a list of category names into the set of classes they cover.

    Tolerant of what a language model or a user might type: singular or plural,
    different capitalisation, and the few obvious synonyms.
    """
    aliases = {
        "person": "people", "human": "people", "humans": "people",
        "pedestrian": "people", "pedestrians": "people",
        "vehicle": "vehicles", "car": "vehicles", "cars": "vehicles",
        "animal": "animals", "pet": "animals", "pets": "animals",
        "furnishing": "furniture", "furnishings": "furniture",
        "outdoor objects": "outdoor", "street furniture": "outdoor",
        "street": "outdoor", "infrastructure": "outdoor",
        "personal item": "personal items", "accessories": "personal items",
        "electronic": "electronics", "appliance": "electronics",
        "appliances": "electronics",
    }

    out: set[str] = set()
    for raw in names:
        key = raw.strip().lower()
        key = aliases.get(key, key)
        out.update(CATEGORIES.get(key, []))
    return out


def category_summary() -> list[dict]:
    """Everything the frontend needs to build its filter chips."""
    return [
        {
            "id": name,
            "label": CATEGORY_LABELS.get(name, name.title()),
            "classes": classes,
            "primary": name in PRIMARY_CATEGORIES,
        }
        for name, classes in CATEGORIES.items()
    ]
