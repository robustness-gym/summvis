from typing import List, Tuple, Union

from htbuilder import span, styles, HtmlElement
from htbuilder.units import px


def hex_to_rgb(hex):
    hex = hex.replace("#", '')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def color_with_opacity(hex_color, opacity):
    rgb = hex_to_rgb(hex_color)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity:.2f})"


def highlight(
        tokens: List[str],
        opacities: List[float],
        colors: List[str],
        **kwargs
):
    return [
        span(
            style=styles(
                background_color=color_with_opacity(color, opacity),
                **kwargs
            )
        )(token) if opacity is not None else token
        for token, opacity, color in zip(tokens, opacities, colors)
    ]


def multi_underline(
        tokens: List[Union[str, HtmlElement]],
        span_groups: List[List[Tuple[int, int, str]]],
        group_colors: List[str],
        underline_thickness=4,
        underline_spacing=1
):
    """Style text with multiple layers of colored underlines.
        Args:
            tokens: list of tokens
            span_groups:
                list of (one for each group)
                    list of (start pos, end pos, id) spans
                where highest level list is associated with a color at each position
            group_colors: list of colors
        Returns:
            list of element elements
    """
    # TODO Add validity checks on spans (non-overlapping, don't exceed length of tokens)
    # IOB encoding =  Inside-Outside-Begin dense encoding of spans for easier processing
    n_groups = len(span_groups)
    n_tokens = len(tokens)
    n_colors = len(group_colors)
    # Initialize IOB encoding to O (Outside)
    iob_encodings = [['O'] * n_tokens for _ in range(n_groups)]  # n_groups x n_tokens
    # Organize span ids by group / token position for easier retrieval later
    span_ids = [[None] * n_tokens for _ in range(n_groups)]  # n_groups x n_tokens
    for group_idx, spans in enumerate(span_groups):
        spans = sorted(spans)
        for span_start, span_end, span_id in spans:
            iob_encodings[group_idx][span_start] = 'B'
            iob_encodings[group_idx][span_start + 1:span_end] = ['I'] * (span_end - span_start - 1)
            span_ids[group_idx][span_start:span_end] = [span_id] * (span_end - span_start)

    # Create list of nested html span elements that will render as multi-underlined text

    elements = []
    underline_slots = [None] * n_groups  # Slots that track which span group is at which underline depth

    for token_idx, token in enumerate(tokens):
        for group_idx in range(n_groups):
            iob = iob_encodings[group_idx][token_idx]
            if iob == 'B' or iob == 'O':
                # Remove group from previous slot if starting new span or outside of a span
                try:
                    slot_idx = underline_slots.index(group_idx)
                except ValueError:
                    pass
                else:
                    underline_slots[slot_idx] = None
        slot_colors = [None] * n_groups  # Color of underline at each depth
        slot_ids = [None] * n_groups  # Associated span id at each underline depth
        for slot_idx, group_idx in enumerate(underline_slots):
            if group_idx is None:
                continue
            slot_colors[slot_idx] = group_colors[group_idx % n_colors]
            slot_ids[slot_idx] = span_ids[group_idx][token_idx]

        # Add underlined space between tokens
        if token_idx > 0:
            elements.append(
                _multi_underline_span('&ensp;', slot_colors, slot_ids, underline_thickness, underline_spacing, False)
            )

        # Set underline slot for any new spans
        for group_idx in range(n_groups):
            iob = iob_encodings[group_idx][token_idx]
            if iob == 'B':
                # Find next available underline slot
                for k in range(len(underline_slots)):
                    if underline_slots[k] is None:
                        underline_slots[k] = group_idx
                        break

        slot_colors = [None] * n_groups  # Color of underline at each depth
        slot_ids = [None] * n_groups  # Associated span id at each underline depth
        for slot_idx, group_idx in enumerate(underline_slots):
            if group_idx is None:
                continue
            slot_colors[slot_idx] = group_colors[group_idx % n_colors]
            slot_ids[slot_idx] = span_ids[group_idx][token_idx]

        elements.append(
            _multi_underline_span(token, slot_colors, slot_ids, underline_thickness, underline_spacing, True)
        )
    return elements


def _multi_underline_span(token, slot_colors, slot_ids, underline_thickness, underline_spacing, is_token):
    element = token
    # Get index of last non-null element
    last_index = None
    for i, color in reversed(list(enumerate(slot_colors))):
        if color is not None:
            last_index = i
            break
    if last_index is None:
        # No active underlines, simply return plain text
        return element
    slot_colors = slot_colors[:last_index + 1]  # Truncate trailing null elements
    for underline_level, (slot_color, slot_id) in enumerate(zip(slot_colors, slot_ids)):
        if slot_color is None:
            color = "rgba(0, 0, 0, 0)"  # Transparent element w/opacity=0
            props = {}
        else:
            color = slot_color
            classes = ["underline"]
            if is_token:
                classes.append("token-underline")
            props = {"data-span-id": slot_id,
                     "class": " ".join(classes)}
        if underline_level == 0:
            padding_bottom = 0
        else:
            padding_bottom = underline_spacing
        display = "inline-block"

        element = span(
            style=styles(
                display=display,
                border_bottom=f"{underline_thickness}px solid",
                border_color=color,
                padding_bottom=px(padding_bottom),
            ),
            **props
        )(element)

    # Return outermost nested span
    return element


if __name__ == "__main__":
    # Test
    text = "The quick brown fox jumps"
    tokens = text.split()
    tokens = [
        "The",
        span(style=styles(color="red"))("quick"),
        "brown",
        "fox",
        "jumps"
    ]
    spans = [
        [(0, 2), (3, 4)],  # green
        [(1, 3)],  # orange
        [(2, 4)],  # blue
    ]
    colors = [
        "#66c2a5",  # green
        "#fc8d62",  # orange
        "#8da0cb",  # blue
    ]

    out = multi_underline(tokens, spans, group_colors=colors)
