# ── Endora — Jetson operations ───────────────────────────────────────────
#
# FOR THE JETSON ONLY. This drives the standalone GPU deployment
# (docker-compose.jetson.yml). The Home Assistant add-on is built and
# updated by the Supervisor and has nothing to do with any of this.
#
# The destructive targets refuse to run anywhere that is not a Jetson, so a
# stray `make prune` on a laptop cannot reach that machine's Docker.
#
#   make            list the targets
#   make rebuild    pull the latest code and restart on it
#   make gpu        confirm inference is actually on the GPU
#   make disk       where the space went
#
COMPOSE := docker compose -f docker-compose.jetson.yml
SERVICE := endora

# Running inside a throwaway container is how the /data targets reach the
# named volume without hard-coding Compose's project-prefixed name for it.
INDATA := $(COMPOSE) run --rm --no-deps --entrypoint sh $(SERVICE) -c

.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps shell build rebuild gpu debug \
        disk prune prune-hard data-size trt-size trt-clear jetson-only

help:  ## List available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Lifecycle ────────────────────────────────────────────────────────────

up:  ## Start the container (builds it if absent)
	$(COMPOSE) up -d

down:  ## Stop and remove the container (the /data volume survives)
	$(COMPOSE) down

restart:  ## Restart without rebuilding — picks up settings.yaml changes
	$(COMPOSE) restart

logs:  ## Follow the log
	$(COMPOSE) logs -f --tail=200 $(SERVICE)

ps:  ## Show container status
	$(COMPOSE) ps

shell:  ## Open a shell in the running container
	$(COMPOSE) exec $(SERVICE) sh

build:  ## Rebuild the image from the current working tree
	$(COMPOSE) build

rebuild:  ## git pull, rebuild, restart — the normal update path
	git pull
	$(COMPOSE) up -d --build
	@echo
	@echo "Give TensorRT a minute if the model changed, then: make gpu"

# ── Checks ───────────────────────────────────────────────────────────────

gpu:  ## Report which execution provider inference actually landed on
	@$(COMPOSE) logs $(SERVICE) 2>/dev/null | grep -m1 'provider=' \
	  || echo "No 'provider=' line yet — the model may still be loading (make logs)."

debug:  ## Print the debug UI URL
	@$(COMPOSE) logs $(SERVICE) 2>/dev/null | grep -m1 'Debug stream:' \
	  || echo "No debug stream — is DEBUG_PORT set in .env?"

# ── Disk ─────────────────────────────────────────────────────────────────
#
# A JetPack base image is several GB and every rebuild leaves the previous
# one behind as an untagged layer set, so this fills a microSD card quickly.
#
# None of these touch volumes. The endora_data volume holds settings.yaml,
# runtime_overrides.yaml, feedback.jsonl and the TensorRT engine cache —
# losing it means re-tuning from scratch, which is why `docker system prune`
# is never invoked here with --volumes.

disk:  ## Show filesystem and Docker space usage
	@df -h / | sed 's/^/  /'
	@echo
	@docker system df

prune:  ## Reclaim space: dangling images + build cache. Keeps volumes.
prune: jetson-only
	docker image prune -f
	docker builder prune -f
	@echo
	@docker system df

prune-hard:  ## Also removes every image not used by a container. Keeps volumes.
prune-hard: jetson-only
	docker image prune -a -f
	docker builder prune -a -f
	@echo "Note: the next build re-pulls the JetPack base image (several GB)."
	@docker system df

data-size:  ## Size of the persistent /data volume, by directory
	@$(INDATA) 'du -sh /data/* 2>/dev/null | sort -h' || true

trt-size:  ## Size of the TensorRT engine cache
	@$(INDATA) 'du -sh /data/trt_cache 2>/dev/null || echo "  (no cache yet)"'

trt-clear:  ## Delete cached TensorRT engines — next start rebuilds them (minutes)
trt-clear: jetson-only
	$(INDATA) 'rm -rf /data/trt_cache/* && echo cleared'
	@echo "Cleared. The next inference will stall while TensorRT rebuilds."

# ── Guard ────────────────────────────────────────────────────────────────

jetson-only:
	@test -f /etc/nv_tegra_release || { \
	  echo "Refusing: this target is for the Jetson, and this is not one."; \
	  echo "(No /etc/nv_tegra_release — that file is what identifies a Tegra board.)"; \
	  exit 1; }
