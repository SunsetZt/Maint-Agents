import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: "/:path*",
        has: [{ type: "host", value: "127.0.0.1" }],
        destination: "http://localhost:3000/:path*",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
