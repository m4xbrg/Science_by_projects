.PHONY: test lint format gallery all
test:
\tpytest -q
lint:
\truff check .
format:
\tblack .
gallery:
\tpython scripts/build_gallery.py
all: format lint test gallery
