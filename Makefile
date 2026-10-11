.PHONY: m1-init m1-up m1-verify m1-status m1-logs m1-down m1-clean static web-check

m1-init:
	./scripts/dev_lite.sh init

m1-up:
	./scripts/dev_lite.sh up

m1-verify:
	./scripts/m1_verify.sh compose-lite

m1-status:
	./scripts/dev_lite.sh status

m1-logs:
	./scripts/dev_lite.sh logs

m1-down:
	./scripts/dev_lite.sh down

m1-clean:
	./scripts/dev_lite.sh clean

static:
	.venv/bin/ruff check src
	.venv/bin/mypy src

web-check:
	web/node_modules/.bin/pnpm --dir web typecheck
	web/node_modules/.bin/pnpm --dir web build
