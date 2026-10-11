import type { Account, NextAuthOptions, Profile, Session } from "next-auth";
import type { JWT } from "next-auth/jwt";
import KeycloakProvider from "next-auth/providers/keycloak";

type IndustrialProfile = Profile & {
  subject_id?: string;
  tenant_id?: string;
  roles?: string[];
};

type KeycloakAccount = Account & {
  refresh_expires_in?: number;
};

type KeycloakRefreshResponse = {
  access_token?: unknown;
  expires_in?: unknown;
  refresh_token?: unknown;
  refresh_expires_in?: unknown;
};

export type IndustrialToken = JWT & {
  apiAccessToken?: string;
  refreshToken?: string;
  subjectId?: string;
  tenantId?: string;
  roles?: string[];
  accessTokenExpiresAt?: number;
  refreshTokenExpiresAt?: number;
  authError?: "RefreshAccessTokenError";
};

export function buildServerToken(
  token: JWT,
  account?: Account | null,
  profile?: Profile,
): IndustrialToken {
  const next: IndustrialToken = { ...token };
  const claims = profile as IndustrialProfile | undefined;
  const keycloakAccount = account as KeycloakAccount | null | undefined;
  if (account?.access_token) next.apiAccessToken = account.access_token;
  if (account?.expires_at) next.accessTokenExpiresAt = account.expires_at;
  if (account?.refresh_token) next.refreshToken = account.refresh_token;
  if (keycloakAccount?.refresh_expires_in) {
    next.refreshTokenExpiresAt =
      Math.floor(Date.now() / 1000) + keycloakAccount.refresh_expires_in;
  }
  if (claims?.subject_id) next.subjectId = claims.subject_id;
  if (claims?.tenant_id) next.tenantId = claims.tenant_id;
  if (Array.isArray(claims?.roles)) next.roles = claims.roles;
  delete next.authError;
  return next;
}

export function projectBrowserSession(
  session: Session,
  token: IndustrialToken,
): Session {
  if (session.user) {
    session.user.subjectId = token.subjectId;
    session.user.tenantId = token.tenantId;
    session.user.roles = token.roles ?? [];
    session.user.accessTokenExpiresAt = token.accessTokenExpiresAt;
  }
  session.authError = token.authError;
  return session;
}

const keycloakIssuer =
  process.env.AUTH_KEYCLOAK_ISSUER ??
  "http://localhost:8080/realms/industrial-ops";
const keycloakInternalIssuer =
  process.env.AUTH_KEYCLOAK_INTERNAL_ISSUER ?? keycloakIssuer;
const keycloakClientId =
  process.env.AUTH_KEYCLOAK_ID ?? "industrial-ops-web";
const keycloakClientSecret = process.env.AUTH_KEYCLOAK_SECRET ?? "";
const ACCESS_TOKEN_REFRESH_BUFFER_SECONDS = 30;

export function accessTokenNeedsRefresh(
  token: IndustrialToken,
  nowSeconds = Math.floor(Date.now() / 1000),
): boolean {
  return (
    typeof token.accessTokenExpiresAt === "number" &&
    token.accessTokenExpiresAt <=
      nowSeconds + ACCESS_TOKEN_REFRESH_BUFFER_SECONDS
  );
}

export async function refreshServerToken(
  token: IndustrialToken,
  fetchImplementation: typeof fetch = globalThis.fetch,
  nowSeconds = Math.floor(Date.now() / 1000),
): Promise<IndustrialToken> {
  if (
    !token.refreshToken ||
    (typeof token.refreshTokenExpiresAt === "number" &&
      token.refreshTokenExpiresAt <= nowSeconds)
  ) {
    return {
      ...token,
      apiAccessToken: undefined,
      authError: "RefreshAccessTokenError",
    };
  }

  const body = new URLSearchParams({
    client_id: keycloakClientId,
    grant_type: "refresh_token",
    refresh_token: token.refreshToken,
  });
  if (keycloakClientSecret) body.set("client_secret", keycloakClientSecret);

  try {
    const response = await fetchImplementation(
      `${keycloakInternalIssuer}/protocol/openid-connect/token`,
      {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
        cache: "no-store",
      },
    );
    const payload = (await response.json()) as KeycloakRefreshResponse;
    if (
      !response.ok ||
      typeof payload.access_token !== "string" ||
      typeof payload.expires_in !== "number"
    ) {
      throw new Error("OIDC token refresh failed");
    }
    return {
      ...token,
      apiAccessToken: payload.access_token,
      accessTokenExpiresAt: nowSeconds + payload.expires_in,
      refreshToken:
        typeof payload.refresh_token === "string"
          ? payload.refresh_token
          : token.refreshToken,
      refreshTokenExpiresAt:
        typeof payload.refresh_expires_in === "number"
          ? nowSeconds + payload.refresh_expires_in
          : token.refreshTokenExpiresAt,
      authError: undefined,
    };
  } catch {
    return {
      ...token,
      apiAccessToken: undefined,
      authError: "RefreshAccessTokenError",
    };
  }
}

const keycloak = KeycloakProvider({
  clientId: keycloakClientId,
  clientSecret: keycloakClientSecret,
  issuer: keycloakIssuer,
});

// The browser needs the host-published authorization URL, while server-side
// token, userinfo, and JWKS calls stay on the Compose network. Keeping the
// public issuer preserves ID-token issuer validation across both paths.
delete keycloak.wellKnown;
keycloak.authorization = {
  url: `${keycloakIssuer}/protocol/openid-connect/auth`,
  params: { scope: "openid email profile" },
};
keycloak.token = `${keycloakInternalIssuer}/protocol/openid-connect/token`;
keycloak.userinfo = `${keycloakInternalIssuer}/protocol/openid-connect/userinfo`;
keycloak.jwks_endpoint = `${keycloakInternalIssuer}/protocol/openid-connect/certs`;

export const authOptions: NextAuthOptions = {
  providers: [{ ...keycloak, checks: ["pkce", "state"] }],
  session: {
    strategy: "jwt",
    maxAge: 60 * 60 * 8,
  },
  pages: {
    signIn: "/login",
    error: "/login",
  },
  callbacks: {
    async jwt({ token, account, profile }) {
      const serverToken = buildServerToken(token, account, profile);
      if (account || !accessTokenNeedsRefresh(serverToken)) {
        return serverToken;
      }
      return refreshServerToken(serverToken);
    },
    async session({ session, token }) {
      return projectBrowserSession(session, token as IndustrialToken);
    },
  },
};
