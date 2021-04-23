from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Union, List, Dict, Optional

from htbuilder import span, div, script, style, link, HtmlElement, styles
from spacy.tokens import Doc

from textstyle import MultiUnderline

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


def highlight(
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


def main_view(
    document: Doc,
    summaries: List[Doc],
    semantic_alignments: Optional[List[Dict]],
    lexical_alignments: Optional[List[Dict]],
    layout: str,
    scroll: bool,
    gray_out_stopwords: bool
):
    # Add document elements

    if document._.name == 'Document':
        document_name = 'Source Document'
    else:
        document_name = document._.name + ' summary'
    doc_header = div(
        id_="document-header"
    )(
        document_name
    )
    doc_elements = []

    # Add document content, which comprises multiple elements, one for each summary. Only the elment corresponding to
    # selected summary will be visible.

    mu = MultiUnderline()

    for summary_idx, summary in enumerate(summaries):
        token_idx_to_sent_idx = {}
        for sent_idx, sent in enumerate(summary.sents):
            for token in sent:
                token_idx_to_sent_idx[token.i] = sent_idx
        is_selected_summary = (summary_idx == 0)  # By default, first summary is selected

        if semantic_alignments is not None:
            doc_token_idx_to_matches = defaultdict(list)
            semantic_alignment = semantic_alignments[summary_idx]
            for summary_token_idx, matches in semantic_alignment.items():
                for doc_token_idx, sim in matches:
                    doc_token_idx_to_matches[doc_token_idx].append((summary_token_idx, sim))
        else:
            doc_token_idx_to_matches = {}

        token_elements = []
        for doc_token_idx, doc_token in enumerate(document):
            if doc_token.is_stop or doc_token.is_punct:
                classes = ["stopword"]
                if gray_out_stopwords:
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
                    el = highlight(
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
        if lexical_alignments is not None:
            lexical_alignment = lexical_alignments[summary_idx]
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
        if scroll:
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
    for summary_idx, summary in enumerate(summaries):
        token_idx_to_sent_idx = {}
        for sent_idx, sent in enumerate(summary.sents):
            for token in sent:
                token_idx_to_sent_idx[token.i] = sent_idx

        spans = []
        matches_ngram = [False] * len(list(summary))
        if lexical_alignments is not None:
            lexical_alignment = lexical_alignments[summary_idx]
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

        if semantic_alignments is not None:
            semantic_alignment = semantic_alignments[summary_idx]
        else:
            semantic_alignment = {}
        token_elements = []
        for token_idx, token in enumerate(summary):
            if token.is_stop or token.is_punct:
                classes = ["stopword"]
                if gray_out_stopwords:
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
                    el = highlight(
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
    if scroll:
        classes.append("scroll")
    if lexical_alignments is not None:
        classes.append("has-lexical-alignment")
    if semantic_alignments is not None:
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
            _class=f"vis-container {layout}-layout"
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
        """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css"
         integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" 
         crossorigin="anonymous">""",
        local_stylesheet(Path(__file__).parent / "summvis.css"),
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
        local_script(Path(__file__).parent / "jquery.color-2.1.2.min.js"),
        local_script(Path(__file__).parent / "summvis.js")
    ]
