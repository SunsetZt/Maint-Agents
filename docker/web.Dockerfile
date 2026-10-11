FROM node:22.23.1-bookworm-slim AS build

ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME:$PATH
WORKDIR /workspace
RUN corepack enable && corepack prepare pnpm@11.20.0 --activate
COPY pnpm-lock.yaml pnpm-workspace.yaml ./
COPY web/package.json ./web/package.json
RUN pnpm install --frozen-lockfile
COPY web ./web
RUN pnpm --dir web build

FROM node:22.23.1-bookworm-slim AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app
WORKDIR /app
COPY --from=build --chown=app:app /workspace/web/.next/standalone ./
COPY --from=build --chown=app:app /workspace/web/.next/static ./web/.next/static
COPY --chown=app:app docker/web-entrypoint.sh /usr/local/bin/web-entrypoint
USER app
EXPOSE 3000
WORKDIR /app/web
ENTRYPOINT ["web-entrypoint"]
CMD ["node", "server.js"]
