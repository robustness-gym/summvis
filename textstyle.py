from collections import defaultdict
from itertools import count
from operator import itemgetter
from typing import List, Tuple, Union

import htbuilder
from htbuilder import styles, HtmlElement
from htbuilder.units import px


def hex_to_rgb(hex):
    hex = hex.replace("#", '')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def color_with_opacity(hex_color, opacity):
    rgb = hex_to_rgb(hex_color)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity:.2f})"


SPACE = "&ensp;"


class MultiUnderline:
    def __init__(
        self,
        underline_thickness=3,
        underline_spacing=1
    ):
        self.underline_thickness = underline_thickness
        self.underline_spacing = underline_spacing

    def markup(
        self,
        tokens: List[Union[str, HtmlElement]],
        spans: List[Tuple[int, int, int, str, str]]
    ):
        """Style text with multiple layers of colored underlines.
            Args:
                tokens: list of tokens, either string or html element
                spans: list of (start_pos, end_pos, rank, color, id) tuples defined as:
                    start_pos: start position of underline span
                    end_pos: end position of underline span
                    rank: rank for stacking order of underlines, all else being equal
                    color: color of underline
                    id: id of underline (encoded as a class label in resulting html element)
            Returns:
                List of HTML elements
        """

        # Map from span start position to span
        start_to_spans = defaultdict(list)
        for span in spans:
            start = span[0]
            start_to_spans[start].append(span)

        # Map from each underline slot position to list of active spans
        slot_to_spans = {}

        # Collection of html elements
        elements = []

        for pos, token in enumerate(tokens):
            # Remove spans that are no longer active (end < pos)
            slot_to_spans = defaultdict(
                list,
                {
                    slot: [span for span in spans if span[1] > pos]  # span[1] contains end of spans
                    for slot, spans in slot_to_spans.items() if spans
                }
            )

            # Add underlines to space between tokens for any continuing underlines
            if pos > 0:
                elements.append(self._get_underline_element(SPACE, slot_to_spans))

            # Find slot for any new spans
            new_spans = start_to_spans.pop(pos, None)
            if new_spans:
                new_spans.sort(key=lambda span: (-(span[1] - span[0]), span[2]))  # Sort by span length (reversed), rank
                for new_span in new_spans:
                    # Find an existing slot or add a new one
                    for slot, spans in sorted(slot_to_spans.items(), key=itemgetter(0)):  # Sort by slot index
                        if spans:
                            containing_span = spans[0]  # The first span in the slot strictly contains all other spans
                            containing_start, containing_end = containing_span[0:2]
                            containing_color = containing_span[3]
                            start, end = new_span[0:2]
                            color = new_span[3]
                            # If the new span (1) is strictly contained in this span, or (2) exactly matches this span
                            # and is the same color, then add span to this slot
                            if end <= containing_end and (
                                (start > containing_start or end < containing_end) or
                                (start == containing_start and end == containing_end and color == containing_color)
                            ):
                                spans.append(new_span)
                                break
                    else:
                        # Find a new slot index to add the span
                        for slot_index in count():
                            spans = slot_to_spans[slot_index]
                            if not spans:  # If slot is free, take it
                                spans.append(new_span)
                                break

            # Add underlines to token for all active spans
            elements.append(self._get_underline_element(token, slot_to_spans))
        return elements

    def _get_underline_element(self, token, slot_to_spans):
        if not slot_to_spans:
            return token
        max_slot_index = max(slot_to_spans.keys())
        element = token
        for slot_index in range(max_slot_index + 1):
            spans = slot_to_spans[slot_index]
            if not spans:
                color = "rgba(0, 0, 0, 0)"  # Transparent element w/opacity=0
                props = {}
            else:
                containing_slot = spans[0]
                color = containing_slot[3]
                classes = ["underline"]
                if token != SPACE:
                    classes.append("token-underline")
                classes.extend([f"span-{span[4]}" for span in spans])  # Encode ids in class names
                props = {
                    "class": " ".join(classes),
                    "data-primary-color": color
                }
            if slot_index == 0:
                padding_bottom = 0
            else:
                padding_bottom = self.underline_spacing
            display = "inline-block"
            element = htbuilder.span(
                style=styles(
                    display=display,
                    border_bottom=f"{self.underline_thickness}px solid",
                    border_color=color,
                    padding_bottom=px(padding_bottom),
                ),
                **props
            )(element)

            # Return outermost nested span
        return element


if __name__ == "__main__":
    from htbuilder import div

    # Test
    text = "The quick brown fox jumps"
    tokens = text.split()
    tokens = [
        "The",
        htbuilder.span(style=styles(color="red"))("quick"),
        "brown",
        "fox",
        "jumps"
    ]
    spans = [
        (0, 2, 0, "green", "green1"),
        (1, 3, 0, "orange", "orange1"),
        (3, 4, 0, "red", "red1"),
        (2, 4, 0, "blue", "blue1"),
        (1, 5, 0, "orange", "orange1"),
    ]

    mu = MultiUnderline()
    html = str(div(mu.markup(tokens, spans)))
    print(html)
