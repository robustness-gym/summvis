$(document).ready(
    function () {

        // Define global variables

        let isDragging = false;
        let saveDragPos;

        let rtime;
        let timeout = false;
        let delta = 200;

        let disableScrollEvent = false;

        let annotateLexical = false;
        let annotateSemantic = false;
        let annotateNovel = false;
        let annotateEntities = false;

        // Define functions

        function clamp(number, min, max) {
            return Math.max(min, Math.min(number, max));
        }

        function hasScroll() {
            const el = $(".display .main-doc");
            return el.prop("scrollHeight") > el.prop("clientHeight");
        }

        function scrollBy(delta) {
            const proxyDoc = $(".display .proxy-doc");
            const proxyScroll = proxyDoc.find(".proxy-scroll");
            const currentTop = parseFloat(proxyScroll.css("top"));
            const newTop = clamp(currentTop + delta, 0, proxyDoc.innerHeight() - proxyScroll.innerHeight());
            proxyScroll.css("top", newTop);
            const mainDoc = $(".display .main-doc");
            const scaleY = mainDoc[0].scrollHeight / proxyDoc.innerHeight();
            mainDoc.scrollTop(newTop * scaleY)
        }

        function getSpanId(el) {
            return getSpanIds(el)[0]
        }

        function getSpanIds(el) {
            return el.attr("class").split(/\s+/).filter(function (x) {
                return x.startsWith("span-")
            });
        }

        function createProxy() {
            const mainDoc = $(".display .main-doc");
            const proxyDoc = $(".display .proxy-doc");
            const proxyHeight = proxyDoc.innerHeight();
            const proxyWidth = proxyDoc.innerWidth();
            const scaleX = 0.8 * proxyWidth / mainDoc.innerWidth();
            const scaleY = proxyHeight / mainDoc[0].scrollHeight;
            const scrollTop = mainDoc.scrollTop();
            const proxyScrollTop = scrollTop * scaleY;
            const proxyScrollBottom = (scrollTop + mainDoc.innerHeight()) * scaleY;
            const proxyScrollHeight = proxyScrollBottom - proxyScrollTop;
            proxyDoc.empty();

            // Loop through underlines in doc view and create associated proxy element
            if (annotateLexical) {
                $(".display .main-doc .token-underline").each(
                    function (index, value) {
                        const el = $(value);
                        const x = el.position().left;
                        const y = mainDoc.scrollTop() + el.position().top - mainDoc.position().top;
                        const newHeight = 3;
                        const color = el.css("border-bottom-color");
                        const proxyPadding = proxyDoc.innerWidth() - proxyDoc.width();
                        const newX = x * scaleX + proxyPadding / 2;
                        const newY = (y + el.height()) * scaleY - newHeight;
                        const newWidth = Math.min(
                            Math.max((el.width() * scaleX) + 1, 5),
                            proxyDoc.width() + proxyPadding / 2 - newX
                        );

                        let classes = "proxy-underline annotation-hidden " + getSpanIds(el).join(" ");
                        const proxyEl = $('<div/>', {
                            "class": classes,
                            "css": {
                                "position": "absolute",
                                "left": Math.round(newX),
                                "top": Math.round(newY),
                                "background-color": color,
                                "width": newWidth,
                                "height": newHeight,
                            }
                        }).appendTo(proxyDoc);
                        proxyEl.data(el.data());
                    }
                );
            }

            // Loop through all active highlights in doc view and create associated proxy element
            if (annotateSemantic) {
                $(".display .main-doc .highlight").each(
                    function (index, value) {
                        const el = $(value);
                        const x = el.position().left;
                        const y = mainDoc.scrollTop() + el.position().top - mainDoc.position().top;
                        const newHeight = 5;
                        const color = el.css("background-color");
                        const proxyPadding = proxyDoc.innerWidth() - proxyDoc.width()
                        const newX = x * scaleX + proxyPadding / 2;
                        const newY = (y + el.height()) * scaleY - newHeight;
                        const newWidth = Math.min(
                            Math.max((el.width() * scaleX) + 1, 5),
                            proxyDoc.width() + proxyPadding / 2 - newX
                        );
                        const proxyEl = $('<div/>', {
                            "class": 'proxy-highlight annotation-hidden',
                            "css": {
                                "position": "absolute",
                                "left": Math.round(newX),
                                "top": Math.round(newY),
                                "background-color": color,
                                "width": newWidth,
                                "height": newHeight,
                            }
                        }).appendTo(proxyDoc);
                        // Copy data attributes
                        proxyEl.data(el.data());
                        // Set classes for matching
                        proxyEl.addClass(el.data("match-classes"))
                    }
                );
            }
            $('<div/>', {
                "class": 'proxy-scroll hidden',
                "css": {
                    "top": proxyScrollTop,
                    "height": proxyScrollHeight,
                }
            }).appendTo(proxyDoc);
            if (hasScroll()) {
                $(".display .proxy-scroll").removeClass("hidden")
            }

            $(".display .proxy-doc")
                .mousedown(function (event) {
                    saveDragPos = parseFloat(event.pageY);
                    isDragging = true;
                    event.preventDefault();
                })
                .mousemove(function (event) {
                    const dragPos = parseFloat(event.pageY);
                    if (isDragging) {
                        const distanceMoved = dragPos - saveDragPos;
                        scrollBy(distanceMoved);
                        saveDragPos = dragPos;
                        event.preventDefault();
                    }
                })
                .mouseup(function (event) {
                    isDragging = false;
                })
                .mouseenter(function () {
                    disableScrollEvent = true;
                    $(".display .proxy-scroll").addClass("hover")
                })
                .mouseleave(function () {
                    isDragging = false;
                    disableScrollEvent = false;
                    $(".display .proxy-scroll").removeClass("hover")
                })
                .on('wheel', function (event) {
                    scrollBy(event.originalEvent.deltaY / 4);
                    event.preventDefault();
                });

            // TODO: Handle user clicking in scroll region

            $(".display .main-doc").scroll(function () {
                if (disableScrollEvent) return;
                $(".display .proxy-scroll")
                    .css(
                        "top", $(this).scrollTop() * scaleY
                    )
            })
        }

        function resizeend() {
            if (new Date() - rtime < delta) {
                setTimeout(resizeend, delta);
            } else {
                timeout = false;
                updateAnnotations();
                toggleScrollbar();
            }
        }

        function toggleScrollbar() {
            if (hasScroll()) {
                $(".display .proxy-scroll").removeClass("hidden");
            } else {
                $(".display .proxy-scroll").addClass("hidden");
            }
        }

        function updateAnnotations() {

            annotateSemantic = $("#option-semantic").hasClass("selected");
            annotateLexical = $("#option-lexical").hasClass("selected");
            annotateEntities = $("#option-entity").hasClass("selected");
            annotateNovel = $("#option-novel").hasClass("selected");

            if (annotateSemantic || annotateLexical) {
                $(".summary-item").addClass("selectable")
            } else {
                $(".summary-item").removeClass("selectable")
            }

            if (annotateLexical) {
                $(".underline").removeClass("annotation-hidden");
                $(".summary-item").addClass("annotate-lexical");
            } else {
                $(".underline").addClass("annotation-hidden");
                $(".summary-item").removeClass("annotate-lexical");
            }
            if (annotateSemantic) {
                $(".highlight").removeClass("annotation-hidden");
            } else {
                $(".highlight").addClass("annotation-hidden");
            }
            if (annotateEntities) {
                $(".summary-item").addClass("annotate-entities")
            } else {
                $(".summary-item").removeClass("annotate-entities")
            }
            if (annotateNovel) {
                $(".summary-item").addClass("annotate-novel")
            } else {
                $(".summary-item").removeClass("annotate-novel")
            }

            createProxy();

            if (annotateLexical) {
                $(".proxy-underline").removeClass("annotation-hidden");
            } else {
                $(".proxy-underline").addClass("annotation-hidden");
            }
            if (annotateSemantic) {
                $(".proxy-highlight").removeClass("annotation-hidden");
            } else {
                $(".proxy-highlight").addClass("annotation-hidden");
            }

            $(".summary-item .highlight").tooltip("disable");
            if (annotateSemantic) {
                $(".summary-item.selected .highlight").tooltip("enable")
            }
        }

        function removeDocTooltips() {
            $("[data-tooltip-timestamp]").tooltip("dispose").removeAttr("data-tooltip-timestamp");
        }

        function resetUnderlines() {
            $('.annotation-invisible').removeClass("annotation-invisible");
            $('.annotation-inactive').removeClass("annotation-inactive");
            $('.temp-underline-color')
                .each(function () {
                    $(this).css("border-color", $(this).data("primary-color"));
                })
                .removeClass("temp-underline-color")
            $('.temp-proxy-underline-color')
                .each(function () {
                    $(this).css("background-color", $(this).data("primary-color"));
                })
                .removeClass("temp-proxy-underline-color")
        }

        function showDocTooltip(el) {
            const topDocHighlightId = $(el).data("top-doc-highlight-id");
            const topDocSim = $(el).data("top-doc-sim");
            const topHighlight = $(`.display .main-doc .highlight[data-highlight-id=${topDocHighlightId}]`);
            if (!isViewable(topHighlight)) {
                return;
            }
            topHighlight.tooltip({title: `Most similar (${topDocSim})`, trigger: "manual", container: "body"});
            topHighlight.tooltip("show");
            const tooltipTimestamp = Date.now();
            // Do not use .data() method to set data attributes as they are not searchable
            topHighlight.attr("data-tooltip-timestamp", tooltipTimestamp);
            setTimeout(function () {
                if (topHighlight.data("tooltip-timestamp") == tooltipTimestamp) {
                    topHighlight.tooltip("dispose").removeAttr("data-tooltip-timestamp");
                }
            }, 8000);
        }

        function highlightUnderlines() {
            const spanId = getSpanId($(this));
            const color = $(this).css("border-bottom-color");
            // TODO Consolidate into single statement
            $(`.summary-item.selected .underline.${spanId}`).removeClass("annotation-inactive");
            $(`.doc .underline.${spanId}`)
                .removeClass("annotation-inactive")
                .each(function () {
                    $(this).css("border-bottom-color", color);
                })
                .addClass("temp-underline-color");
            $(`.proxy-underline.${spanId}`)
                .removeClass("annotation-inactive")
                .each(function () {
                    $(this).css("background-color", color);
                })
                .addClass("temp-proxy-underline-color");

            $(`.summary-item.selected .underline:not(.${spanId})`).addClass("annotation-inactive");
            $(`.doc .underline:not(.${spanId})`).addClass("annotation-inactive");
            $(`.proxy-underline:not(.${spanId})`).addClass("annotation-inactive");

            $(".summary-item.selected .highlight:not(.annotation-hidden)").addClass("annotation-inactive");
        }

        function resetHighlights() {
            removeDocTooltips();
            $('.summary-item.selected .annotation-inactive').removeClass("annotation-inactive");
            $('.summary-item.selected .annotation-invisible').removeClass("annotation-invisible");
            $('.temp-highlight-color')
                .each(function () {
                    $(this).css("background-color", $(this).data("primary-color"));
                })
                .removeClass("temp-highlight-color");
            $('.highlight.selected').removeClass("selected");
            $('.proxy-highlight.selected').removeClass("selected");
            $('.summary-item [title]').removeAttr("title");
        }

        function highlightToken() {
            const highlightId = $(this).data("highlight-id");
            $(`.summary-item.selected .highlight:not(.summary-highlight-${highlightId})`).addClass("annotation-inactive");
            $('.highlight.selected').removeClass("selected")
            $('.proxy-highlight.selected').removeClass("selected")
            const matchedDocHighlight = `.display .main-doc .summary-highlight-${highlightId}`;
            const matchedProxyHighlight = `.proxy-doc .summary-highlight-${highlightId}`;
            $(matchedDocHighlight + ", " + matchedProxyHighlight)
                .each(function () {
                    const newHighlightColor = $(this).data(`color-${highlightId}`);
                    $(this).css("background-color", newHighlightColor);
                    $(this).addClass("selected");
                })
                .addClass("temp-highlight-color");
            $(".underline").addClass("annotation-inactive");
            $(".proxy-underline").addClass("annotation-invisible")
            showDocTooltip(this);
            $(this).addClass("selected");
            $(this).removeClass("annotation-inactive");
            $('.summary-item [title]').removeAttr("title");
            if (!isViewable($(matchedDocHighlight))) {
                $(this).attr("title", "Click to scroll to most similar word.")
            }
        }

        function isViewable(el) {
            const elTop = el.offset().top;
            const elBottom = elTop + el.outerHeight();
            const scrollRegion = $(".display .main-doc");
            const scrollTop = scrollRegion.offset().top;
            const scrollBottom = scrollTop + scrollRegion.outerHeight();
            return elTop > scrollTop && elBottom < scrollBottom;
        }

        // Initialization

        $(function () {
            $('[data-toggle="tooltip"]').tooltip({
                // 'boundary': '.summary-container'
                trigger: 'hover'
            })
        })
        updateAnnotations();

        // Bind events

        $(window).resize(function () {
            rtime = new Date();
            if (timeout === false) {
                timeout = true;
                setTimeout(resizeend, delta);
            }
        });

        $(".summary-list").on(
            "click",
            ".summary-item.selectable:not(.selected)",
            function () {
                const summary_index = $(this).data("index");

                // Update summary items
                $(".summary-item.selected").removeClass("selected")
                $(this).addClass("selected")

                // Update doc
                // Show the version of document aligned with selected summary index
                $(`.doc[data-index=${summary_index}]`).removeClass("nodisplay").addClass("display");
                // Hide the version of document not aligned with selected summary index
                $(`.doc[data-index!=${summary_index}]`).removeClass("display").addClass("nodisplay");

                updateAnnotations();
            }
        );

        $("#option-lexical").click(function () {
            $(this).toggleClass("selected")
            updateAnnotations()
        });
        $("#option-semantic").click(function () {
            $(this).toggleClass("selected")
            updateAnnotations()
        });
        $("#option-novel").click(function () {
            $(this).toggleClass("selected")
            updateAnnotations()
        });
        $("#option-entity").click(function () {
            $(this).toggleClass("selected")
            updateAnnotations()
        });

        const activeUnderlines = ".summary-item.selected .underline:not(.annotation-inactive):not(.annotation-hidden)";
        $(".summary-list").on(
            "mouseenter",
            activeUnderlines,
            function () {
                highlightUnderlines.call(this);
            }
        );

        $(".summary-list").on(
            "mouseleave",
            activeUnderlines,
            resetUnderlines
        );
        $(".summary-list").on(
            "click",
            activeUnderlines,
            function () {
                // Find aligned underline in doc  and scroll doc to that position
                highlightUnderlines.call(this);
                const mainDoc = $(".display .main-doc");
                const spanId = getSpanId($(this));
                const matchedUnderline = $(`.doc .underline.${spanId}`);
                mainDoc.animate({
                        scrollTop: mainDoc.scrollTop() +
                            matchedUnderline.offset().top - mainDoc.offset().top - 60
                    },
                    300
                )
            }
        );

        const activeHighlights = ".summary-item.selected .highlight:not(.annotation-hidden):not(.matches-ngram), " +
            ".summary-item.selected:not(.annotate-lexical) .highlight:not(.annotation-hidden)";
        $(".summary-list").on(
            "mouseenter",
            activeHighlights,
            function () {
                highlightToken.call(this);
            })
        $(".summary-list").on(
            "mouseleave",
            activeHighlights,
            function () {
                resetHighlights();
                resetUnderlines();
            }
        );
        $(".summary-list").on(
            "click",
            activeHighlights,
            function () {
                highlightToken.call(this);
                // Find corresponding highlight in doc representing max similarity and scroll doc to that position
                const topDocHighlightId = $(this).data("top-doc-highlight-id");
                removeDocTooltips(topDocHighlightId);
                const topDocHighlight = $(`.display .main-doc .highlight[data-highlight-id=${topDocHighlightId}]`);
                const mainDoc = $(".display .main-doc");
                const el = this;
                mainDoc.animate({
                        scrollTop: mainDoc.scrollTop() +
                            topDocHighlight.offset().top - mainDoc.offset().top - 60
                    },
                    300,
                    function () {
                        setTimeout(
                            function () {
                                // If no other tooltips have since been displayed
                                if ($("[data-tooltip-timestamp]").length == 0) {
                                    showDocTooltip(el);
                                } else {
                                    console.log("Not showing tooltip because one already exists")
                                }
                            },
                            100
                        )
                    }
                )
            }
        );
        $(".summary-list").on(
            "mouseleave",
            ".summary-item.selected .content",
            function () {
                resetHighlights();
                resetUnderlines();
            },
        );
    }
);

