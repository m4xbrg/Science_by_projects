PY=python
validate: ; $(PY) scripts/validate_meta.py
test: ; pytest -q
smoke: ; pytest -q tests/test_smoke_runtime.py
gallery: ; $(PY) scripts/build_gallery.py
new:
	@echo 'Usage: make new name=slug title="Title" domain=physics math=ode'
	@test "$(name)" != "" || (echo "Missing name=slug"; exit 1)
	$(PY) tools/new_project.py $(name) --title "$(title)" --domain "$(domain)" --math-core "$(math)"
	$(PY) scripts/repair_meta.py --apply
	$(PY) scripts/validate_meta.py
	pytest -q
