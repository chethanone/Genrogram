"use client";

import Navbar from "../../components/Navbar";
import Link from "next/link";
import {
  AudioWaveform,
  SlidersHorizontal,
  Waves,
  RotateCcw,
} from "lucide-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useAudioAnalysis } from "../context/AudioAnalysisContext";

const parameters = [
  ["Sample rate", "22,050 Hz", "Audio sampling frequency"],
  ["Mel bands", "128", "Frequency representation"],
  ["FFT size", "2,048", "Frequency resolution"],
  ["Hop length", "512", "Time-step between frames"],
  ["Segment", "3.0 sec", "Model input duration"],
];

export default function SpectrogramLab() {
  const [imageExpanded, setImageExpanded] = useState(false);
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 767px)");
    const update = () => setIsSmallScreen(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);
  const {
    file,
    audioUrl,
    spectrogramUrl,
    onsetSeconds,
    processing,
    error,
    clearTrack,
  } = useAudioAnalysis();

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#F4F1EE] text-[#161616]">
      <Navbar />

      <section className="mx-auto max-w-7xl px-6 pb-20 pt-14 lg:px-10 lg:pt-20">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-3xl"
        >
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 py-2 font-mono text-xs uppercase tracking-wider text-[#77716E]">
            <Waves size={13} />
            Spectrogram Lab
          </div>

          <h1 className="text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl">
            See the music
            <span className="block text-[#C96F70]">as an image.</span>
          </h1>

          <p className="mt-6 max-w-2xl text-lg leading-8 text-[#77716E]">
            The spectrogram is generated automatically when you classify a
            track. Navigate here without uploading the song again.
          </p>
        </motion.div>

        {!file ? (
          <div className="mt-12 rounded-3xl border border-black/10 bg-white p-12 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-[#161616] text-white">
              <AudioWaveform size={25} />
            </div>

            <h2 className="mt-6 text-2xl font-bold">No track loaded</h2>

            <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[#77716E]">
              Upload and classify a track on the Classify page. Its
              spectrogram will then be available here automatically.
            </p>

            <Link
              href="/"
              className="mt-7 inline-flex rounded-xl bg-[#161616] px-5 py-3.5 text-sm font-semibold text-white transition-colors hover:bg-[#C96F70]"
            >
              Go to Classify
            </Link>
          </div>
        ) : (
          <>
            {/* Shared track + model configuration */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12 grid min-w-0 items-stretch gap-6 lg:grid-cols-2"
            >
              {/* Current track */}
              <div className="flex h-[340px] w-full min-w-0 flex-col rounded-3xl border border-black/10 bg-white p-6 shadow-[0_20px_60px_rgba(22,22,22,0.05)]">
                <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                  Current track
                </div>

                <div className="mt-5 flex flex-1 flex-col rounded-2xl bg-[#F4F1EE] p-5">
                  <div
                    className="truncate font-semibold"
                    title={file.name}
                  >
                    {file.name}
                  </div>

                  {audioUrl && (
                    <audio
                      className="mt-5 block w-full max-w-full"
                      controls
                      src={audioUrl}
                    />
                  )}

                  {onsetSeconds !== null && (
                    <div className="mt-5 rounded-xl bg-white px-4 py-3 text-sm text-[#77716E]">
                      <span className="font-semibold text-[#161616]">
                        Music detected at
                      </span>{" "}
                      approximately {onsetSeconds.toFixed(2)}s.
                    </div>
                  )}
                </div>

                <button
                  onClick={clearTrack}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-black/10 bg-white px-5 py-3 font-mono text-xs uppercase tracking-wider text-[#77716E] transition-colors hover:text-[#161616]"
                >
                  <RotateCcw size={14} />
                  Clear current track
                </button>
              </div>

              {/* Model configuration */}
              <div className="flex h-[560px] w-full min-w-0 flex-col rounded-3xl bg-[#161616] p-6 text-white">
                <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-white/50">
                  <SlidersHorizontal size={14} />
                  Model input configuration
                </div>

                <div className="mt-5 grid flex-1 grid-cols-2 gap-3">
                  {parameters.map(([label, value, description]) => (
                    <div
                      key={label}
                      className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
                    >
                      <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/40">
                        {label}
                      </div>

                      <div className="mt-2 text-2xl font-semibold tracking-tight">
                        {value}
                      </div>

                      <div className="mt-1 text-xs leading-5 text-white/45">
                        {description}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-3 shrink-0 rounded-2xl border border-[#C96F70]/30 bg-[#C96F70]/10 p-4">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#E5A0A1]">
                    Processing pipeline
                  </div>

                  <div className="mt-2 break-words text-sm leading-6 text-white/65">
                    Audio{" "}
                    <span className="text-white/30">|</span> onset detection{" "}
                    <span className="text-white/30">|</span> 3-second view{" "}
                    <span className="text-white/30">|</span> Mel-spectrogram{" "}
                    <span className="text-white/30">|</span> normalization
                  </div>
                </div>
              </div>
            </motion.section>

            {error && (
              <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
                {error}
              </div>
            )}

            {processing && (
              <div className="mt-6 rounded-2xl border border-black/10 bg-white px-5 py-4 text-sm text-[#77716E]">
                Generating the shared analysis. This runs once for the
                uploaded track.
              </div>
            )}

            {/* Spectrogram */}
            {spectrogramUrl && (
              <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 rounded-3xl border border-black/10 bg-white p-6 shadow-[0_20px_60px_rgba(22,22,22,0.05)] lg:p-8"
              >
                <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
                  <div>
                    <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                      Visualization
                    </div>

                    <h2 className="mt-2 text-2xl font-bold tracking-tight">
                      Mel-frequency representation
                    </h2>

                    <p className="mt-2 max-w-2xl text-sm leading-6 text-[#77716E]">
                      The displayed 3-second window begins at the detected
                      musical onset, so leading silence is not shown as an
                      empty block.
                    </p>
                  </div>

                  <div className="shrink-0 font-mono text-xs uppercase tracking-wider text-[#77716E]">
                    128 MEL BANDS <span className="text-black/20">|</span> 3 SEC
                  </div>
                </div>

                <div className="mt-7 overflow-hidden rounded-2xl bg-[#171717] p-4">
                  <button type="button" onClick={() => isSmallScreen && setImageExpanded(true)} className="block w-full" aria-label={isSmallScreen ? "Expand Mel spectrogram" : undefined}>
                    <img
                      src={spectrogramUrl}
                      alt="Mel spectrogram aligned to musical onset"
                      className="mx-auto block w-full object-contain"
                    />
                  </button>
                </div>

                {isSmallScreen && imageExpanded && (
                  <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#161616]/95 p-3" role="dialog" aria-modal="true" aria-label="Expanded Mel spectrogram" onClick={() => setImageExpanded(false)}>
                    <button type="button" className="absolute right-4 top-4 z-10 flex h-10 w-10 items-center justify-center rounded-full bg-white text-xl font-medium text-[#161616] shadow-lg" onClick={() => setImageExpanded(false)} aria-label="Close expanded spectrogram">×</button>
                    <img src={spectrogramUrl} alt="Expanded Mel spectrogram aligned to musical onset" className="max-h-[94vh] max-w-[96vw] object-contain" onClick={(event) => event.stopPropagation()} />
                  </div>
                )}

                <div className="mt-4 flex justify-between font-mono text-[10px] uppercase tracking-wider text-[#AAA4A0]">
                  <span>Low frequency</span>
                  <span>Time from onset</span>
                  <span>Higher frequency</span>
                </div>
              </motion.section>
            )}
          </>
        )}
      </section>

      <footer className="border-t border-black/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-8 font-mono text-xs text-[#77716E] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span>GENGROGRAM / SPECTROGRAM LAB</span>
          <span>22,050 HZ | 128 MEL BANDS</span>
        </div>
      </footer>
    </main>
  );
}