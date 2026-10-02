"use client";
import Navbar from "../components/Navbar";
import { useRef, useState } from "react";
import { useAudioAnalysis } from "./context/AudioAnalysisContext";
import Link from "next/link";
import {
  Upload,
  Music2,
  Sparkles,
  Loader2,
  RotateCcw,
  BarChart3,
} from "lucide-react";
import { motion } from "framer-motion";

type PredictionItem = {
  genre: string;
  probability: number;
};

type PredictionResult = {
  genre: string;
  confidence: number;
  probabilities: Record<string, number>;
  top_predictions: PredictionItem[];
  segment_count: number;
};

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);

  const { file, audioUrl, result, spectrogramUrl, processing: isClassifying, error, selectFile, processTrack, clearTrack } = useAudioAnalysis();
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = (selectedFile: File) => {
    selectFile(selectedFile);
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const selectedFile = event.target.files?.[0];

    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  const handleDrop = (
    event: React.DragEvent<HTMLDivElement>
  ) => {
    event.preventDefault();
    setIsDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  };

  const classifyAudio = async () => {
    await processTrack();
  };

  const reset = () => {
    clearTrack();
    if (inputRef.current) inputRef.current.value = "";
  };

  const formatGenre = (genre: string) => {
    return genre.charAt(0).toUpperCase() + genre.slice(1);
  };

  return (
    <main className="min-h-screen bg-[#F4F1EE] text-[#161616]">
      {/* Header */}
      <Navbar />

      {/* Hero */}
      <section className="mx-auto max-w-7xl px-6 pb-16 pt-16 lg:px-10 lg:pt-24">
        <div className="grid gap-14 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 py-2 font-mono text-xs uppercase tracking-wider text-[#77716E]"
            >
              <Sparkles size={14} />
              <div className="flex items-center gap-2">
                <span>CNN</span>

                <span
                  className="h-1.5 w-1.5 rounded-full bg-[#D9A441]"
                  aria-hidden="true"
                />

                <span>MEL SPECTROGRAM</span>
              </div>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="max-w-3xl text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl lg:text-7xl"
            >
              What does your
              <span className="block text-[#C96F70]">
                music sound like?
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.16 }}
              className="mt-7 max-w-xl text-lg leading-8 text-[#77716E]"
            >
              Upload a track and let GENGROGRAM analyze its Mel-spectrogram
              representation using a convolutional neural network trained
              across 20 musical genres.
            </motion.p>
          </div>

          {/* Upload card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="relative"
          >
            <div className="absolute -right-3 -top-3 h-20 w-20 rounded-full border border-[#C96F70]/30" />

            <div className="relative h-[600px] rounded-3xl border border-black/10 bg-white p-5 shadow-[0_20px_60px_rgba(22,22,22,0.08)]">
              <div
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`flex h-[300px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 text-center transition ${isDragging
                  ? "border-[#C96F70] bg-[#C96F70]/5"
                  : "border-black/10 hover:border-[#C96F70]/50 hover:bg-[#F4F1EE]/50"
                  }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
                  onChange={handleInputChange}
                  className="hidden"
                />

                <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-[#161616] text-white">
                  <Upload size={25} />
                </div>

                <h2 className="text-xl font-semibold">
                  Drop your track here
                </h2>

                <p className="mt-2 text-sm text-[#77716E]">
                  or click to browse your computer
                </p>

                <div className="mt-6 font-mono text-[10px] uppercase tracking-[0.2em] text-[#AAA4A0]">
                  WAV | MP3 | FLAC | OGG | M4A
                </div>
              </div>

              {file && (
                <div className="mt-4 h-[250px] rounded-2xl bg-[#F4F1EE] p-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[#C96F70] text-white">
                      <Music2 size={18} />
                    </div>

                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">
                        {file.name}
                      </p>

                      <p className="font-mono text-xs text-[#77716E]">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>

                    <button
                      onClick={reset}
                      className="rounded-full p-2 text-[#77716E] transition hover:bg-white hover:text-[#161616]"
                      aria-label="Remove audio"
                    >
                      <RotateCcw size={17} />
                    </button>
                  </div>

                  {audioUrl && (
                    <audio
                      className="mt-4 w-full"
                      controls
                      src={audioUrl}
                    />
                  )}

                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      classifyAudio();
                    }}
                    disabled={isClassifying}
                    className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-[#161616] px-5 py-3.5 font-semibold text-white transition hover:bg-[#C96F70] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isClassifying ? (
                      <>
                        <Loader2 size={17} className="animate-spin" />
                        Analyzing track...
                      </>
                    ) : (
                      <>
                        <Sparkles size={17} />
                        Classify Genre
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-auto mt-8 max-w-3xl rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-20"
          >
            <div className="mb-8 flex items-end justify-between gap-5">
              <div>
                <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                  Classification result
                </div>

                <h2 className="mt-2 text-3xl font-bold tracking-tight">
                  Your track reads as
                </h2>
              </div>

              <div className="hidden font-mono text-xs text-[#77716E] sm:block">
                {result.segment_count} segments analyzed
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
              {/* Main prediction */}
              <div className="rounded-3xl bg-[#161616] p-8 text-white">
                <div className="font-mono text-xs uppercase tracking-[0.2em] text-white/45">
                  Detected genre
                </div>

                <div className="mt-10 text-5xl font-bold tracking-tight">
                  {formatGenre(result.genre)}
                </div>

                <div className="mt-10">
                  <div className="flex items-end justify-between">
                    <span className="font-mono text-xs uppercase tracking-wider text-white/45">
                      Confidence
                    </span>

                    <span className="text-3xl font-semibold text-[#E5A0A1]">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>

                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{
                        width: `${result.confidence * 100}%`,
                      }}
                      transition={{ duration: 0.8 }}
                      className="h-full rounded-full bg-[#C96F70]"
                    />
                  </div>
                </div>
              </div>

              {/* Top predictions */}
              <div className="rounded-3xl border border-black/10 bg-white p-8">
                <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                  <BarChart3 size={14} />
                  Probability distribution
                </div>

                <div className="mt-8 space-y-6">
                  {result.top_predictions.map((prediction, index) => (
                    <div key={prediction.genre}>
                      <div className="mb-2 flex items-center justify-between">
                        <span className="font-medium">
                          <span className="mr-3 font-mono text-xs text-[#AAA4A0]">
                            0{index + 1}
                          </span>
                          {formatGenre(prediction.genre)}
                        </span>

                        <span className="font-mono text-xs text-[#77716E]">
                          {(prediction.probability * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className="h-2 overflow-hidden rounded-full bg-[#F4F1EE]">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{
                            width: `${Math.max(
                              prediction.probability * 100,
                              1
                            )}%`,
                          }}
                          transition={{
                            duration: 0.7,
                            delay: index * 0.1,
                          }}
                          className={`h-full rounded-full ${index === 0
                            ? "bg-[#C96F70]"
                            : "bg-[#161616]/30"
                            }`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Spectrogram */}
            {spectrogramUrl && (
              <div className="mt-6 rounded-3xl border border-black/10 bg-white p-6">
                <div className="mb-5">
                  <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                    Spectrogram
                  </div>

                  <h3 className="mt-1 text-xl font-semibold">
                    Mel-frequency representation
                  </h3>
                </div>

                <div className="overflow-hidden rounded-2xl bg-[#171717] p-4">
                  <img
                    src={spectrogramUrl}
                    alt="Mel spectrogram of the uploaded track"
                    className="mx-auto block max-h-[300px] w-full object-contain"
                  />
                </div>
              </div>
            )}
          </motion.section>
        )}
      </section>

      {/* Footer */}
      <footer className="border-t border-black/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-8 font-mono text-xs text-[#77716E] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span>GENGROGRAM / MUSIC GENRE CLASSIFICATION</span>
          <span>
            <div className="flex items-center gap-3">
              <span
                className="relative h-3 w-3 rounded-full bg-[#D9A441]"
                aria-hidden="true"
              >
                <span className="absolute inset-[-3px] rounded-full border border-[#D9A441]/40" />
              </span>

              <span
                className="relative h-2.5 w-2.5 rounded-full bg-[#E2BD68]"
                aria-hidden="true"
              >
                <span className="absolute inset-[-3px] rounded-full border border-[#D9A441]/30" />
              </span>

              <span
                className="relative h-3 w-3 rounded-full bg-[#D9A441]"
                aria-hidden="true"
              >
                <span className="absolute inset-[-3px] rounded-full border border-[#D9A441]/40" />
              </span>
            </div>
          </span>
        </div>
      </footer>
    </main>
  );
}
