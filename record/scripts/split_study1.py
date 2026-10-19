#!/usr/bin/env python3
"""Split the frozen Study 1 monograph (``monograph/main.tex``) into chapter sub-documents.

The Study 1 text is carried over verbatim; only three mechanical changes are made:
table inputs point at ``tables/study1/``, labels and references gain an ``s1:`` prefix
so they cannot collide with the new parts, and the front matter becomes a chapter.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "monograph" / "main.tex"
OUT = ROOT / "record" / "study1"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]


def adapt(text: str) -> str:
    text = text.replace(r"\input{tables/", r"\input{tables/study1/")
    text = re.sub(r"\\label\{", r"\\label{s1:", text)
    text = re.sub(r"\\(ref|Cref|cref|pageref|autoref|eqref)\{([^}]*)\}",
                  lambda m: "\\" + m.group(1) + "{" + ",".join("s1:" + k.strip() for k in m.group(2).split(",")) + "}",
                  text)
    return text


def main() -> None:
    source = SOURCE.read_text()
    body = source[source.index(r"\chapter*{Document status and scope}"):source.index(r"\bibliographystyle")]
    front, rest = body.split(r"\tableofcontents", 1)
    rest = rest.split(r"\pagenumbering{arabic}", 1)[1]
    main_part, appendix = rest.split(r"\appendix", 1)
    OUT.mkdir(parents=True, exist_ok=True)
    front = front.replace(r"\chapter*{Document status and scope}", r"\chapter{Study 1 record: status, scope, and abstract}")
    front = front.replace(r"\addcontentsline{toc}{chapter}{Document status and scope}", "")
    front = front.replace(r"\begin{abstract}", r"\begin{keybox}{Abstract of the Study 1 record (13 September 2026)}")
    front = front.replace(r"\end{abstract}", r"\end{keybox}")
    files = [("ch00_status_scope", front)]
    chapters = re.split(r"(?=\\chapter\{)", main_part)
    for index, chunk in enumerate(c for c in chapters if c.strip().startswith(r"\chapter{")):
        title = re.match(r"\\chapter\{([^}]*)\}", chunk.strip()).group(1)
        files.append((f"ch{index + 1:02d}_{slug(title)}", chunk))
    appendices = re.split(r"(?=\\chapter\{)", appendix)
    for index, chunk in enumerate(c for c in appendices if c.strip().startswith(r"\chapter{")):
        title = re.match(r"\\chapter\{([^}]*)\}", chunk.strip()).group(1)
        files.append((f"app{index + 1:02d}_{slug(title)}", chunk))
    names = []
    for name, chunk in files:
        # Mechanical addition: a label on every chapter heading so later parts can cite it.
        chunk = re.sub(r"(\\chapter\{[^}]*\})", lambda m: m.group(1) + "\\label{ch:" + name + "}", chunk, count=1)
        (OUT / f"{name}.tex").write_text("% Carried over verbatim from monograph/main.tex (Study 1 record, 13 Sep 2026).\n"
                                         + adapt(chunk).strip() + "\n")
        names.append(name)
    (OUT / "index.txt").write_text("\n".join(names) + "\n")
    print(len(names), "files:", ", ".join(names))


if __name__ == "__main__":
    main()
