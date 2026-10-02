import type { Metadata } from 'next';
import './globals.css';
import { AudioAnalysisProvider } from './context/AudioAnalysisContext';

export const metadata: Metadata = {
  title: 'GENROGRAM — Music Genre Classification',
  description: 'Develop a Music Genre Classification System Using Spectrogram and CNN',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#F4F1EE] text-[#161616] antialiased min-h-screen">
        <AudioAnalysisProvider>{children}</AudioAnalysisProvider>
      </body>
    </html>
  );
}
