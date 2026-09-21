import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BhuVistaar - Satellite Super-Resolution Dashboard",
  description: "AI-Powered Super Resolution Mapping",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#09090b] text-[#f4f4f5] antialiased">
        {children}
      </body>
    </html>
  );
}
