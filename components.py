from collections import defaultdict
from itertools import count
from operator import itemgetter
from pathlib import Path
from typing import Dict, Optional
from typing import List, Tuple, Union

import htbuilder
import streamlit as st
from htbuilder import span, div, script, style, link, styles, HtmlElement, br
from htbuilder.units import px
from spacy.tokens import Doc

palette = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
inactive_color = "#BBB"


def local_stylesheet(path):
    with open(path) as f:
        css = f.read()
    return style()(
        css
    )


def remote_stylesheet(url):
    return link(
        href=url
    )


def local_script(path):
    with open(path) as f:
        code = f.read()
    return script()(
        code
    )


def remote_script(url):
    return script(
        src=url
    )


def get_color(sent_idx):
    return palette[sent_idx % len(palette)]


def hex_to_rgb(hex):
    hex = hex.replace("#", '')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def color_with_opacity(hex_color, opacity):
    rgb = hex_to_rgb(hex_color)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{opacity:.2f})"


class Component:

    def show(self, width=None, height=None, scrolling=True, **kwargs):
        out = div(style=styles(
            **kwargs
        ))(self.html())
        html = str(out)
        st.components.v1.html(html, width=width, height=height, scrolling=scrolling)

    def html(self):
        raise NotImplemented


class MainView(Component):

    def __init__(
        self,
        document: Doc,
        summaries: List[Doc],
        semantic_alignments: Optional[List[Dict]],
        lexical_alignments: Optional[List[Dict]],
        layout: str,
        scroll: bool,
        gray_out_stopwords: bool
    ):
        self.document = document
        self.summaries = summaries
        self.semantic_alignments = semantic_alignments
        self.lexical_alignments = lexical_alignments
        self.layout = layout
        self.scroll = scroll
        self.gray_out_stopwords = gray_out_stopwords

    def html(self):

        # Add document elements
        if self.document._.name == 'Document':
            document_name = 'Source Document'
        else:
            document_name = self.document._.name + ' summary'
        doc_header = div(
            id_="document-header"
        )(
            document_name
        )
        doc_elements = []

        # Add document content, which comprises multiple elements, one for each summary. Only the elment corresponding to
        # selected summary will be visible.

        mu = MultiUnderline()

        for summary_idx, summary in enumerate(self.summaries):
            token_idx_to_sent_idx = {}
            for sent_idx, sent in enumerate(summary.sents):
                for token in sent:
                    token_idx_to_sent_idx[token.i] = sent_idx
            is_selected_summary = (summary_idx == 0)  # By default, first summary is selected

            if self.semantic_alignments is not None:
                doc_token_idx_to_matches = defaultdict(list)
                semantic_alignment = self.semantic_alignments[summary_idx]
                for summary_token_idx, matches in semantic_alignment.items():
                    for doc_token_idx, sim in matches:
                        doc_token_idx_to_matches[doc_token_idx].append((summary_token_idx, sim))
            else:
                doc_token_idx_to_matches = {}

            token_elements = []
            for doc_token_idx, doc_token in enumerate(self.document):
                if doc_token.is_stop or doc_token.is_punct:
                    classes = ["stopword"]
                    if self.gray_out_stopwords:
                        classes.append("grayed-out")
                    el = span(
                        _class=" ".join(classes)
                    )(
                        doc_token.text
                    )

                else:
                    matches = doc_token_idx_to_matches.get(doc_token_idx)
                    if matches:
                        summary_token_idx, sim = max(matches, key=itemgetter(1))
                        sent_idx = token_idx_to_sent_idx[summary_token_idx]
                        color_primary = get_color(sent_idx)
                        highlight_color_primary = color_with_opacity(color_primary, sim)
                        props = {
                            'data-highlight-id': str(doc_token_idx),
                            'data-primary-color': highlight_color_primary
                        }
                        match_classes = []
                        for summary_token_idx, sim in matches:
                            sent_idx = token_idx_to_sent_idx[summary_token_idx]
                            match_classes.append(f"summary-highlight-{summary_idx}-{summary_token_idx}")
                            color = color_with_opacity(get_color(sent_idx), sim)
                            props[f"data-color-{summary_idx}-{summary_token_idx}"] = color
                        props["data-match-classes"] = " ".join(match_classes)
                        el = self._highlight(
                            doc_token.text,
                            highlight_color_primary,
                            color_primary,
                            match_classes + ["annotation-hidden"],
                            **props
                        )
                    else:
                        el = doc_token.text
                token_elements.append(el)

            spans = []
            if self.lexical_alignments is not None:
                lexical_alignment = self.lexical_alignments[summary_idx]
                for summary_span, doc_spans in lexical_alignment.items():
                    summary_span_start, summary_span_end = summary_span
                    span_id = f"{summary_idx}-{summary_span_start}-{summary_span_end}"
                    sent_idx = token_idx_to_sent_idx[summary_span_start]
                    for doc_span_start, doc_span_end in doc_spans:
                        spans.append((
                            doc_span_start,
                            doc_span_end,
                            sent_idx,
                            get_color(sent_idx),
                            span_id
                        ))
            token_elements = mu.markup(token_elements, spans)

            classes = ["main-doc", "bordered"]
            if self.scroll:
                classes.append("scroll")

            main_doc = div(
                _class=" ".join(classes)
            )(
                token_elements
            ),

            classes = ["doc"]
            if is_selected_summary:
                classes.append("display")
            else:
                classes.append("nodisplay")
            doc_elements.append(
                div(
                    **{
                        "class": " ".join(classes),
                        "data-index": summary_idx
                    }
                )(
                    main_doc,
                    div(_class="proxy-doc"),
                    div(_class="proxy-scroll")
                )
            )

        summary_title = "Summary"
        summary_header = div(
            id_="summary-header"
        )(
            summary_title,
            div(id="summary-header-gap"),
        )

        summary_items = []
        for summary_idx, summary in enumerate(self.summaries):
            token_idx_to_sent_idx = {}
            for sent_idx, sent in enumerate(summary.sents):
                for token in sent:
                    token_idx_to_sent_idx[token.i] = sent_idx

            spans = []
            matches_ngram = [False] * len(list(summary))
            if self.lexical_alignments is not None:
                lexical_alignment = self.lexical_alignments[summary_idx]
                for summary_span in lexical_alignment.keys():
                    start, end = summary_span
                    matches_ngram[slice(start, end)] = [True] * (end - start)
                    span_id = f"{summary_idx}-{start}-{end}"
                    sent_idx = token_idx_to_sent_idx[start]
                    spans.append((
                        start,
                        end,
                        sent_idx,
                        get_color(sent_idx),
                        span_id
                    ))

            if self.semantic_alignments is not None:
                semantic_alignment = self.semantic_alignments[summary_idx]
            else:
                semantic_alignment = {}
            token_elements = []
            for token_idx, token in enumerate(summary):
                if token.is_stop or token.is_punct:
                    classes = ["stopword"]
                    if self.gray_out_stopwords:
                        classes.append("grayed-out")
                    el = span(
                        _class=" ".join(classes)
                    )(
                        token.text
                    )
                else:
                    classes = []
                    if token.ent_iob_ in ('I', 'B'):
                        classes.append("entity")
                    if matches_ngram[token_idx]:
                        classes.append("matches-ngram")
                    matches = semantic_alignment.get(token_idx)
                    if matches:
                        top_match = max(matches, key=itemgetter(1))
                        top_sim = max(top_match[1], 0)
                        top_doc_token_idx = top_match[0]
                        props = {
                            "data-highlight-id": f"{summary_idx}-{token_idx}",
                            "data-top-doc-highlight-id": str(top_doc_token_idx),
                            "data-top-doc-sim": f"{top_sim:.2f}",
                        }
                        classes.extend([
                            "annotation-hidden",
                            f"summary-highlight-{summary_idx}-{token_idx}"
                        ])
                        sent_idx = token_idx_to_sent_idx[token_idx]
                        el = self._highlight(
                            token.text,
                            color_with_opacity(get_color(sent_idx), top_sim),
                            color_with_opacity(get_color(sent_idx), 1),
                            classes,
                            **props
                        )
                    else:
                        if classes:
                            el = span(_class=" ".join(classes))(token.text)
                        else:
                            el = token.text
                token_elements.append(el)

            token_elements = mu.markup(token_elements, spans)

            classes = ["summary-item"]
            if summary_idx == 0:  # Default is for first summary to be selected
                classes.append("selected")

            summary_items.append(
                div(
                    **{"class": ' '.join(classes), "data-index": summary_idx}
                )(
                    div(_class="name")(summary._.name),
                    div(_class="content")(token_elements)
                )
            )
        classes = ["summary-list", "bordered"]
        if self.scroll:
            classes.append("scroll")
        if self.lexical_alignments is not None:
            classes.append("has-lexical-alignment")
        if self.semantic_alignments is not None:
            classes.append("has-semantic-alignment")
        summary_list = div(
            _class=" ".join(classes)
        )(
            summary_items
        )

        annotation_key = \
            """
              <ul class="annotation-key">
                <li class="annotation-key-label">Annotations:</li>
                <li id="option-lexical" class="option selected">
                    <span class="annotation-key-ngram">N-Gram overlap</span>
                </li>
                <li id="option-semantic" class="option selected">
                    <span class="annotation-key-semantic">Semantic overlap</span>
                </li>
                <li id="option-novel" class="option selected">
                    <span class="annotation-key-novel">Novel words</span>
                </li>
                <li id="option-entity" class="option selected">
                    <span class="annotation-key-entity">Novel entities</span>
                </li>
        
            </ul>
            """

        body = div(
            annotation_key,
            div(
                _class=f"vis-container {self.layout}-layout"
            )(
                div(
                    _class="doc-container"
                )(
                    doc_header,
                    *doc_elements
                ),
                div(
                    _class="summary-container"
                )(
                    summary_header,
                    summary_list
                )
            ),
        )
        return [
            """<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">""",
            local_stylesheet(Path(__file__).parent / "resources" / "summvis.css"),
            """<link rel="preconnect" href="https://fonts.gstatic.com">
                <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">""",
            body,
            """<script
                src="https://code.jquery.com/jquery-3.5.1.min.js"
                integrity="sha256-9/aliU8dGd2tb6OSsuzixeV4y/faTqgFtohetphbbj0="
                crossorigin="anonymous"></script>
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"
                 integrity="sha384-Piv4xVNRyMGpqkS2by6br4gNJ7DXjqk09RmUpJ8jgGtD7zP9yug3goQfGII0yAns"
                  crossorigin="anonymous"></script>""",
            local_script(Path(__file__).parent / "resources" / "jquery.color-2.1.2.min.js"),
            local_script(Path(__file__).parent / "resources" / "summvis.js"),
            """<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-gtEjrD/SeCtmISkJkNUaaKMoLD0//ElJ19smozuHV6z3Iehds+3Ulb9Bn9Plx0x4" crossorigin="anonymous"></script>"""
        ]

    def _highlight(
        self,
        token: Union[str, HtmlElement],
        background_color,
        dotted_underline_color,
        classes: List[str],
        **props
    ):
        return span(
            _class=" ".join(classes + ["highlight"]),
            style=styles(
                background_color=background_color,
                border_bottom=f"4px dotted {dotted_underline_color}",
            ),
            **props
        )(token)


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

        first_token_in_line = True
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
            if first_token_in_line:
                first_token_in_line = False
            else:
                elements.append(self._get_underline_element(SPACE, slot_to_spans))

            # Find slot for any new spans
            new_spans = start_to_spans.pop(pos, None)
            if new_spans:
                new_spans.sort(
                    key=lambda span: (-(span[1] - span[0]), span[2]))  # Sort by span length (reversed), rank
                for new_span in new_spans:
                    # Find an existing slot or add a new one
                    for slot, spans in sorted(slot_to_spans.items(), key=itemgetter(0)):  # Sort by slot index
                        if spans:
                            containing_span = spans[
                                0]  # The first span in the slot strictly contains all other spans
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
            if token in ("\n", "\r", "\r\n"):
                elements.append(br())
                first_token_in_line = True
            else:
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
