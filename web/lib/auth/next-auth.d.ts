import "next-auth";

declare module "next-auth" {
  interface Session {
    authError?: "RefreshAccessTokenError";
    user?: {
      name?: string | null;
      email?: string | null;
      image?: string | null;
      subjectId?: string;
      tenantId?: string;
      roles: string[];
      accessTokenExpiresAt?: number;
    };
  }
}
