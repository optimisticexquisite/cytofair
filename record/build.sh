#!/usr/bin/env bash
# Build the full lifecycle record and one standalone PDF per part.
set -euo pipefail
cd "$(dirname "$0")"
tectonic --keep-intermediates main.tex
mkdir -p parts_pdf
for part in p1_overview p2_study1 p3_design p4_engineering p5_results p6_followup p7_synthesis p8_appendices; do
  cat > "sub_${part}.tex" <<TEX
\documentclass[11pt,oneside]{report}
\usepackage{xr-hyper}
\input{preamble}
\externaldocument{main}
\begin{document}
\tableofcontents
\clearpage
\input{parts/${part}}
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
TEX
  tectonic "sub_${part}.tex" && mv "sub_${part}.pdf" "parts_pdf/${part}.pdf"
  rm -f "sub_${part}.tex"
done
mv main.pdf cytofair_lifecycle_record.pdf
