from bs4 import BeautifulSoup, NavigableString

def parse_post_body(html: str) -> str:
    """
    Given the innerHTML of <div class="s-prose js-post-body">…</div>,
    return one string per paragraph (separated by two newlines),
    where each paragraph interleaves:
      • plain text (outside any <script> tags), and
      • raw LaTeX code from <script type="math/tex*">…</script> tags.

    This skips any MathJax‐rendered Unicode, capturing only the underlying
    <script> content for formulas.
    """
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = []

    # For each <p> tag in the post-body…
    for p_tag in soup.find_all("p"):
        pieces = []

        # Loop over each direct child of <p>
        for node in p_tag.children:
            # 1) If it's plain text…
            if isinstance(node, NavigableString):
                pieces.append(node)

            # 2) If it's a <script> whose type starts with "math/tex"…
            elif getattr(node, "name", None) == "script" and node.get("type", "").startswith("math/tex"):
                raw_latex = node.string or ""
                pieces.append(raw_latex.strip())

            else:
                # 3) Otherwise, it’s some other Tag (e.g. <span class="math-container">…</span>).
                #    Check if there are ANY <script type="math/tex"> inside this subtree.
                math_scripts = node.find_all(
                    "script",
                    attrs={"type": lambda t: t and t.startswith("math/tex")}
                )
                if math_scripts:
                    # Append the raw LaTeX from each <script> found, in order
                    for sc in math_scripts:
                        pieces.append((sc.string or "").strip())
                    # Skip any visible text inside this node, because it’s just MathJax markup
                    continue
                else:
                    # No math scripts here, so append whatever visible text exists
                    visible = node.get_text()
                    if visible:
                        pieces.append(visible)

        paragraph_text = "".join(pieces).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    # Join paragraphs with two newlines (you can change to "\n" if you prefer single-line breaks)
    return "\n\n".join(paragraphs)

def parse_title_mixed(html: str) -> str:
    """
    Given the innerHTML of <a class="s-link">…</a> for a question title,
    return a single string that interleaves plain text and raw LaTeX in
    exactly the order they appeared. However, any MathJax preview/rendered
    spans (class contains "MathJax") are skipped entirely.

    Example: 
      HTML:  <span class="MathJax_Preview">C1</span>
             <span class="MathJax">C¹</span>
             <script type="math/tex">C^1</script>
             " function "
             <span class="MathJax_Preview">f>0</span>
             <span class="MathJax">𝑓>0</span>
             <script type="math/tex">f>0</script>
             …
      Returns: "C^1 function f>0 …"
    """
    soup = BeautifulSoup(html, "html.parser")
    pieces = []

    for node in soup.contents:
        # 1) If it's plain text, append directly
        if isinstance(node, NavigableString):
            # strip leading/trailing spaces; if non-empty, keep it
            text = node.strip()
            if text:
                pieces.append(text)

        # 2) If it's a <script type="math/tex*">, grab raw LaTeX
        elif getattr(node, "name", None) == "script" and node.get("type", "").startswith("math/tex"):
            raw_latex = (node.string or "").strip()
            if raw_latex:
                pieces.append(raw_latex)

        else:
            # Check if this tag has any "MathJax" class; if so, skip it entirely
            cls = node.get("class", []) or []
            # If any class name starts with "MathJax", do NOT append its text:
            if any(c.startswith("MathJax") for c in cls):
                continue

            # Otherwise, look for nested <script type="math/tex*"> inside it.
            nested_scripts = node.find_all(
                "script",
                attrs={"type": lambda t: t and t.startswith("math/tex")}
            )
            if nested_scripts:
                # Append raw LaTeX from each nested <script>
                for sc in nested_scripts:
                    raw = (sc.string or "").strip()
                    if raw:
                        pieces.append(raw)
            else:
                # No nested <script> and no MathJax class ⇒ grab visible text
                visible = node.get_text().strip()
                if visible:
                    pieces.append(visible)

    # Join with spaces to preserve separation
    return " ".join(pieces).strip()