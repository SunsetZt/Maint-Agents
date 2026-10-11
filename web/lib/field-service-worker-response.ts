export const FIELD_SERVICE_WORKER_SOURCE = String.raw`"use strict";

const CACHE_PREFIX = "ioap-field-shell-";
const CACHE_NAME = CACHE_PREFIX + "v1";
const OFFLINE_URL = "/field-offline.html";
const PRECACHE_URLS = Object.freeze([
  OFFLINE_URL,
  "/manifest.webmanifest",
  "/icon.svg",
]);
const STATIC_EXACT_PATHS = new Set(PRECACHE_URLS);
const PROTECTED_PATH_PREFIXES = Object.freeze([
  "/api/",
  "/bff/",
  "/auth/",
  "/oidc/",
  "/oauth/",
  "/_next/data/",
  "/_next/image",
  "/media/",
  "/audio/",
  "/video/",
  "/speech/",
  "/realtime/",
  "/objects/",
  "/object-store/",
  "/downloads/",
  "/uploads/",
]);

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS)),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names
          .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
          .map((name) => caches.delete(name)),
      ))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("message", (event) => {
  const message = event.data;
  if (
    message !== null
    && typeof message === "object"
    && Object.keys(message).length === 1
    && message.type === "SKIP_WAITING"
  ) {
    self.skipWaiting();
  }
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);

  if (
    request.method !== "GET"
    || url.origin !== self.location.origin
    || isProtectedRequest(request, url)
  ) {
    return;
  }

  if (isFieldNavigation(request, url)) {
    event.respondWith(fieldNavigation(request));
    return;
  }

  if (isApprovedStaticPath(url.pathname)) {
    event.respondWith(cacheFirstStatic(request, url));
  }
});

function isFieldNavigation(request, url) {
  return request.mode === "navigate"
    && (url.pathname === "/field" || url.pathname.startsWith("/field/"));
}

function isApprovedStaticPath(pathname) {
  return STATIC_EXACT_PATHS.has(pathname) || pathname.startsWith("/_next/static/");
}

function isProtectedRequest(request, url) {
  const accept = (request.headers.get("accept") || "").toLowerCase();
  const authorization = request.headers.get("authorization");
  const rsc = request.headers.get("rsc");
  return Boolean(authorization)
    || rsc === "1"
    || url.searchParams.has("_rsc")
    || accept.includes("text/event-stream")
    || request.destination === "audio"
    || request.destination === "video"
    || PROTECTED_PATH_PREFIXES.some((prefix) => url.pathname.startsWith(prefix));
}

async function fieldNavigation(request) {
  try {
    return await fetch(request);
  } catch {
    const cache = await caches.open(CACHE_NAME);
    const fallback = await cache.match(OFFLINE_URL);
    return fallback || Response.error();
  }
}

async function cacheFirstStatic(request, url) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheableStaticResponse(response, url)) {
    await cache.put(request, response.clone());
  }
  return response;
}

function isCacheableStaticResponse(response, url) {
  const cacheControl = (response.headers.get("cache-control") || "").toLowerCase();
  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  const disposition = (response.headers.get("content-disposition") || "").toLowerCase();
  const exactShellAsset = STATIC_EXACT_PATHS.has(url.pathname);
  return response.ok
    && response.type !== "opaque"
    && !cacheControl.includes("private")
    && !cacheControl.includes("no-store")
    && !contentType.includes("text/event-stream")
    && !disposition.includes("attachment")
    && (exactShellAsset || url.pathname.startsWith("/_next/static/"));
}
`;

export function GET(): Response {
  return new Response(FIELD_SERVICE_WORKER_SOURCE, {
    status: 200,
    headers: {
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Content-Type": "application/javascript; charset=utf-8",
      Pragma: "no-cache",
      "Service-Worker-Allowed": "/field/",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
