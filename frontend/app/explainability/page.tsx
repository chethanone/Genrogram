"use client";

import Navbar from "../../components/Navbar";
import Link from "next/link";
import {
  FileAudio,
  Flame,
  Info,
  RotateCcw,
} from "lucide-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { useAudioAnalysis } from "../context/AudioAnalysisContext";

const steps = [
  [
    "01",
    "Use the Classify result",
    "The already-computed genre is used as the Grad-CAM target.",
  ],
  [
    "02",
    "Find a contributing segment",
    "The segment with the strongest target-class response is selected.",
  ],
  [
    "03",
    "Trace gradients",
    "Gradients from Conv4 weight the learned feature maps.",
  ],
  [
    "04",
    "Overlay the heatmap",
    "The activation map is aligned with the selected Mel-spectrogram.",
  ],
];

export default function ExplainabilityPage() {
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
    gradcamUrl,
    result,
    gradcamSegmentIndex,
    processing,
    error,
    clearTrack,
  } = useAudioAnalysis();

  const genre = result?.genre;
  const confidence = result?.confidence ?? null;

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
            <Flame size={13} />
            Grad-CAM Explainability
          </div>

          <h1 className="text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl">
            See where the
            <span className="block text-[#C96F70]">
              CNN is looking.
            </span>
          </h1>

          <p className="mt-6 max-w-2xl text-lg leading-8 text-[#77716E]">
            Grad-CAM uses the same classification result generated on the
            Classify page. It explains a contributing 3-second segment instead
            of independently predicting another genre.
          </p>
        </motion.div>

        {!file ? (
          <div className="mt-12 rounded-3xl border border-black/10 bg-white p-12 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-[#161616] text-white">
              <FileAudio size={25} />
            </div>

            <h2 className="mt-6 text-2xl font-bold">No track loaded</h2>

            <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[#77716E]">
              Upload and classify a track first. The explanation is generated
              automatically and stays available while you navigate.
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
            {/* Shared track + Grad-CAM process */}
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-12 grid min-w-0 items-stretch gap-6 lg:grid-cols-2"
            >
              {/* Current track */}
              <div className="flex h-[320px] min-w-0 flex-col rounded-3xl border border-black/10 bg-white p-6 shadow-[0_20px_60px_rgba(22,22,22,0.05)]">
                <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                  Current track
                </div>

                <div className="mt-5 flex flex-1 flex-col rounded-2xl bg-[#F4F1EE] p-5">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#C96F70] text-white">
                      <FileAudio size={18} />
                    </div>

                    <div className="min-w-0">
                      <p
                        className="truncate font-medium"
                        title={file.name}
                      >
                        {file.name}
                      </p>

                      <p className="font-mono text-xs text-[#77716E]">
                        Shared analysis track
                      </p>
                    </div>
                  </div>

                  {audioUrl && (
                    <audio
                      className="mt-5 w-full"
                      controls
                      src={audioUrl}
                    />
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

              {/* Grad-CAM process */}
              <div className="flex h-[500px] min-w-0 flex-col rounded-3xl bg-[#161616] p-6 text-white">
                <div className="font-mono text-xs uppercase tracking-[0.2em] text-white/50">
                  How Grad-CAM works
                </div>

                <div className="mt-5 grid flex-1 grid-cols-2 gap-3">
                  {steps.map(([number, title, description]) => (
                    <div
                      key={number}
                      className="rounded-2xl border border-white/10 bg-white/[0.04] p-4"
                    >
                      <div className="font-mono text-xs text-[#C96F70]">
                        {number}
                      </div>

                      <h3 className="mt-2 text-base font-semibold">
                        {title}
                      </h3>

                      <p className="mt-2 text-xs leading-5 text-white/45">
                        {description}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="mt-3 rounded-2xl border border-[#C96F70]/30 bg-[#C96F70]/10 p-4">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-[#E5A0A1]">
                    Target layer
                  </div>

                  <div className="mt-2 text-sm text-white/70">
                    Conv4 convolutional feature maps
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
                Generating the shared explanation once for this track.
              </div>
            )}

            {/* Grad-CAM result */}
            {gradcamUrl && (
              <motion.section
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 rounded-3xl border border-black/10 bg-white p-6 shadow-[0_20px_60px_rgba(22,22,22,0.05)] lg:p-8"
              >
                <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
                  <div>
                    <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                      Explanation result
                    </div>

                    <h2 className="mt-2 text-2xl font-bold tracking-tight">
                      CNN activation map
                    </h2>

                    <p className="mt-2 text-sm text-[#77716E]">
                      Contributing 3-second segment{" "}
                      <span className="text-black/20">|</span> Conv4 Grad-CAM
                      {gradcamSegmentIndex !== null
                        ? ` | Segment ${gradcamSegmentIndex + 1}`
                        : ""}
                    </p>
                  </div>

                  {genre && (
                    <div className="rounded-2xl bg-[#161616] px-5 py-4 text-white">
                      <div className="font-mono text-[10px] uppercase tracking-wider text-white/45">
                        Same predicted genre
                      </div>

                      <div className="mt-1 flex items-end gap-3">
                        <span className="text-2xl font-bold">
                          {genre}
                        </span>

                        {confidence !== null && (
                          <span className="font-mono text-xs text-[#E5A0A1]">
                            {(confidence * 100).toFixed(2)}%
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="mt-7 overflow-hidden rounded-2xl bg-[#171717] p-4">
                  <button type="button" onClick={() => isSmallScreen && setImageExpanded(true)} className="block w-full" aria-label={isSmallScreen ? "Expand Grad-CAM spectrogram" : undefined}>
                    <img
                      src={gradcamUrl}
                      alt="Grad-CAM explanation over Mel-spectrogram"
                      className="mx-auto block w-full object-contain"
                    />
                  </button>
                </div>

                {isSmallScreen && imageExpanded && (
                  <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#161616]/95 p-3" role="dialog" aria-modal="true" aria-label="Expanded Grad-CAM spectrogram" onClick={() => setImageExpanded(false)}>
                    <button type="button" className="absolute right-4 top-4 z-10 flex h-10 w-10 items-center justify-center rounded-full bg-white text-xl font-medium text-[#161616] shadow-lg" onClick={() => setImageExpanded(false)} aria-label="Close expanded Grad-CAM">×</button>
                    <img src={gradcamUrl} alt="Expanded Grad-CAM explanation over Mel-spectrogram" className="max-h-[94vh] max-w-[96vw] object-contain" onClick={(event) => event.stopPropagation()} />
                  </div>
                )}

                <div className="mt-5 flex flex-col gap-3 rounded-2xl bg-[#F4F1EE] p-5 sm:flex-row sm:items-start">
                  <Info
                    size={18}
                    className="mt-0.5 shrink-0 text-[#C96F70]"
                  />

                  <p className="text-sm leading-6 text-[#77716E]">
                    Warmer regions indicate stronger Grad-CAM activation for
                    the same genre selected by the classifier. The
                    visualization shows model activation, not a human-readable
                    explanation of musical content.
                  </p>
                </div>
              </motion.section>
            )}
          </>
        )}
      </section>

      <footer className="border-t border-black/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-8 font-mono text-xs text-[#77716E] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span>GENGROGRAM / EXPLAINABILITY</span>
          <span>GRAD-CAM | CONV4</span>
        </div>
      </footer>
    </main>
  );
}