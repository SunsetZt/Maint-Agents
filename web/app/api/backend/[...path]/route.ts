import { getToken } from "next-auth/jwt";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

type RouteContext = { params: Promise<{ path: string[] }> };

const FORWARDED_REQUEST_HEADERS = [
  "content-type",
  "idempotency-key",
  "if-match",
  "x-content-sha256",
  "last-event-id",
  "x-request-id",
  "x-emergency-grant-id",
] as const;
const FORWARDED_RESPONSE_HEADERS = [
  "cache-control",
  "content-disposition",
  "content-type",
  "etag",
  "x-content-type-options",
  "x-ag-ui-profile",
  "x-request-id",
  "x-trace-id",
] as const;
const ACCESS_TOKEN_REFRESH_BUFFER_SECONDS = 30;

async function proxy(request: NextRequest, context: RouteContext) {
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  });
  const accessToken =
    token && typeof token.apiAccessToken === "string"
      ? token.apiAccessToken
      : undefined;
  if (!accessToken) {
    return NextResponse.json(
      {
        error: {
          category: "authentication",
          code: "authentication_required",
          message: "Authentication required",
          retryable: false,
        },
      },
      { status: 401 },
    );
  }
  const accessTokenExpiresAt =
    token && typeof token.accessTokenExpiresAt === "number"
      ? token.accessTokenExpiresAt
      : undefined;
  if (
    accessTokenExpiresAt !== undefined &&
    accessTokenExpiresAt <=
      Math.floor(Date.now() / 1000) + ACCESS_TOKEN_REFRESH_BUFFER_SECONDS
  ) {
    return NextResponse.json(
      {
        error: {
          category: "authentication",
          code: "access_token_expired",
          message: "Session refresh required",
          retryable: true,
        },
      },
      { status: 401 },
    );
  }

  const { path } = await context.params;
  const baseUrl = process.env.API_INTERNAL_URL ?? "http://api:8000";
  const upstream = new URL(`/${path.map(encodeURIComponent).join("/")}`, baseUrl);
  upstream.search = request.nextUrl.search;
  const headers = new Headers({ Authorization: `Bearer ${accessToken}` });
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const response = await fetch(upstream, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    cache: "no-store",
    redirect: "manual",
    // Required by Node fetch for streamed request bodies.
    duplex: "half",
  } as RequestInit & { duplex: "half" });
  const responseHeaders = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = response.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return new NextResponse(response.body, {
    status: response.status,
    headers: responseHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
