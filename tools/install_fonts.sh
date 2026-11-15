#!/usr/bin/env bash
# Installs open (OFL) fonts with Cyrillic support. Variable fonts are instanced to
# static weights so that Pango (Manim Text), XeLaTeX (fontspec) and libass all
# resolve the same face deterministically.
set -euo pipefail
DEST=${1:-/usr/share/fonts/truetype/factory}
mkdir -p "$DEST"
TMP=$(mktemp -d)
GF=https://github.com/google/fonts/raw/main/ofl
fetch() { curl -fsSL --retry 3 -o "$TMP/$1" "$2"; }
fetch caveat.ttf      "$GF/caveat/Caveat%5Bwght%5D.ttf"
fetch montserrat.ttf  "$GF/montserrat/Montserrat%5Bwght%5D.ttf"
fetch nunito.ttf      "$GF/nunito/Nunito%5Bwght%5D.ttf"
inst() { fonttools varLib.instancer "$TMP/$1" "wght=$2" --update-name-table -q -o "$DEST/$3"; }
inst caveat.ttf 400 Caveat-Regular.ttf
inst caveat.ttf 700 Caveat-Bold.ttf
inst montserrat.ttf 800 Montserrat-ExtraBold.ttf
inst montserrat.ttf 900 Montserrat-Black.ttf
inst nunito.ttf 900 Nunito-Black.ttf
rm -rf "$TMP"
fc-cache -f >/dev/null
fc-list | grep -i "$DEST" || true
