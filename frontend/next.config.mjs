const isExport = process.env.NEXT_OUTPUT_EXPORT === "true";

/** @type {import('next').NextConfig} */
const nextConfig = {
  ...(isExport ? { output: "export" } : {}),
  images: {
    unoptimized: true,
  },
  ...(!isExport
    ? {
        async rewrites() {
          const backendUrl = (
            process.env.BACKEND_URL ||
            process.env.NEXT_PUBLIC_API_URL ||
            "http://127.0.0.1:8000"
          ).replace(/\/+$/, "");
          return [
            {
              source: "/api/:path*",
              destination: `${backendUrl}/api/:path*`,
            },
          ];
        },
      }
    : {}),
};

export default nextConfig;
