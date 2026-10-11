import { getServerSession } from "next-auth";

import { authOptions } from "@/lib/auth/config";

export { authOptions };

export function auth() {
  return getServerSession(authOptions);
}
