import type { Metadata } from 'next';
import { IBM_Plex_Sans, JetBrains_Mono, Plus_Jakarta_Sans } from 'next/font/google';
import './globals.css';

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['500', '600', '700'],
  variable: '--font-display',
});

const ibmPlexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-body',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono',
});

export const metadata: Metadata = {
  title: 'QH4-FieldPermutation · Vector Stage & Verification Workbench',
  description:
    'QH4-FieldPermutation v0.2.0 interactive vector transformation visualizer, mathematical verification suite, and Pelagos multi-GPU adapter workbench over F_65537.',
  openGraph: {
    title: 'QH4-FieldPermutation · Vector Stage & Verification Workbench',
    description:
      'Interactive forward/inverse vector stage visualizer, M01-M10 verification suite, and isolated Python/Triton reference architecture over F_65537.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'QH4-FieldPermutation · Vector Stage & Verification Workbench',
    description:
      'Interactive forward/inverse vector stage visualizer, M01-M10 verification suite, and isolated Python/Triton reference architecture over F_65537.',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${plusJakarta.variable} ${ibmPlexSans.variable} ${jetbrainsMono.variable}`}>
      <body
        className="min-h-screen bg-[#F8FAFC] text-[#0F172A] antialiased selection:bg-[#0F4C81]/15 selection:text-[#0F4C81]"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
