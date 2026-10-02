"use client";

import Navbar from "../../components/Navbar";
import {
  ArrowDown,
  CheckCircle2,
  Cpu,
  Layers3,
  Network,
  Settings2,
  Sparkles,
} from "lucide-react";
import { motion } from "framer-motion";

const blocks = [
  {
    number: "01",
    title: "Convolution Block 1",
    detail: "1 → 32 channels",
    description: "3×3 convolution · BatchNorm · ReLU · MaxPool",
  },
  {
    number: "02",
    title: "Convolution Block 2",
    detail: "32 → 64 channels",
    description: "3×3 convolution · BatchNorm · ReLU · MaxPool",
  },
  {
    number: "03",
    title: "Convolution Block 3",
    detail: "64 → 128 channels",
    description: "3×3 convolution · BatchNorm · ReLU · MaxPool",
  },
  {
    number: "04",
    title: "Convolution Block 4",
    detail: "128 → 256 channels",
    description: "3×3 convolution · BatchNorm · ReLU",
  },
  {
    number: "05",
    title: "Global Average Pooling",
    detail: "256 → 1×1",
    description: "Spatial feature aggregation",
  },
  {
    number: "06",
    title: "Dropout",
    detail: "p = 0.30",
    description: "Regularization before classification",
  },
  {
    number: "07",
    title: "Output Layer",
    detail: "256 → 20 classes",
    description: "Final genre logits",
  },
];

const genres = [
  "Blues",
  "Classical",
  "Country",
  "Disco",
  "Hip-Hop",
  "Jazz",
  "Metal",
  "Pop",
  "Reggae",
  "Rock",
  "Amapiano",
  "Hyperpop",
  "K-Pop",
  "Phonk",
  "Techno",
  "Bollywood",
  "Desi Hip-Hop",
  "Haryanvi",
  "I-Pop",
  "Punjabi Pop",
];

const trainingDetails = [
  ["Sample rate", "22,050 Hz"],
  ["Segment duration", "3 seconds"],
  ["Mel bands", "128"],
  ["FFT / hop", "2048 / 512"],
  ["Optimizer", "AdamW"],
  ["Learning rate", "0.001"],
  ["Weight decay", "0.0001"],
  ["Batch size", "64"],
  ["Dropout", "0.30"],
  ["Selection metric", "Validation Macro F1"],
];

const architectureComparison = [
  {
    name: "CustomCNN",
    params: "393,940",
    accuracy: "75.54%",
    f1: "75.58%",
    status: "Production",
  },
  {
    name: "MobileNetV3-Small",
    params: "1,538,068",
    accuracy: "74.92%",
    f1: "75.73%",
    status: "Compared",
  },
];

export default function ModelPage() {
  return (
    <main className="min-h-screen bg-[#F4F1EE] text-[#161616]">
      <Navbar />

      <section className="mx-auto max-w-7xl px-6 pb-20 pt-14 lg:px-10 lg:pt-20">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-3xl"
        >
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 py-2 font-mono text-xs uppercase tracking-wider text-[#77716E]">
            <Network size={13} />
            Model Architecture
          </div>

          <h1 className="text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl">
            Inside the
            <span className="block text-[#C96F70]">classifier.</span>
          </h1>

          <p className="mt-6 max-w-2xl text-lg leading-8 text-[#77716E]">
            GENGROGRAM uses a compact custom convolutional neural network to
            learn musical patterns from Mel-spectrogram representations and
            classify tracks across 20 genre categories.
          </p>
        </motion.div>

        {/* Key stats */}
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Architecture", "CustomCNN"],
            ["Parameters", "393,940"],
            ["Input", "1 × 128 × 130"],
            ["Classes", "20 genres"],
          ].map(([label, value], index) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="rounded-3xl border border-black/10 bg-white p-6"
            >
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#77716E]">
                {label}
              </div>

              <div className="mt-3 text-2xl font-bold">{value}</div>
            </motion.div>
          ))}
        </div>

        {/* Architecture */}
        <section className="mt-8 rounded-3xl bg-[#161616] p-6 text-white lg:p-8">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-white/45">
            <Layers3 size={14} />
            Network architecture
          </div>

          <div className="mx-auto mt-8 max-w-2xl">
            {/* Input */}
            <div className="rounded-2xl border border-[#C96F70]/40 bg-[#C96F70]/10 p-5 text-center">
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#E5A0A1]">
                Input
              </div>

              <div className="mt-2 text-xl font-semibold">
                Mel Spectrogram
              </div>

              <div className="mt-1 font-mono text-xs text-white/45">
                1 × 128 × 130
              </div>
            </div>

            <div className="flex justify-center py-3">
              <ArrowDown size={20} className="text-white/30" />
            </div>

            {blocks.map((block, index) => (
              <div key={block.number}>
                <div className="rounded-2xl border border-white/10 bg-white/[0.045] px-4 py-3.5 transition hover:border-[#C96F70]/40">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-start gap-4">
                      <div className="font-mono text-xs text-[#C96F70]">
                        {block.number}
                      </div>

                      <div>
                        <div className="text-base font-semibold">
                          {block.title}
                        </div>

                        <div className="mt-1 text-sm text-white/45">
                          {block.description}
                        </div>
                      </div>
                    </div>

                    <div className="font-mono text-xs text-white/55">
                      {block.detail}
                    </div>
                  </div>
                </div>

                {index < blocks.length - 1 && (
                  <div className="flex justify-center py-2">
                    <ArrowDown size={18} className="text-white/25" />
                  </div>
                )}
              </div>
            ))}

            <div className="flex justify-center py-3">
              <ArrowDown size={20} className="text-white/30" />
            </div>

            {/* Output */}
            <div className="rounded-2xl border border-[#C96F70]/40 bg-[#C96F70]/10 p-5 text-center">
              <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#E5A0A1]">
                Output
              </div>

              <div className="mt-2 text-xl font-semibold">
                20 Genre Probabilities
              </div>

              <div className="mt-1 font-mono text-xs text-white/45">
                Softmax prediction
              </div>
            </div>
          </div>
        </section>

        {/* Classes + configuration */}
        <section className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-black/10 bg-white p-7">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
              <Sparkles size={14} />
              Classification classes
            </div>

            <div className="mt-7 grid grid-cols-2 gap-3 sm:grid-cols-3">
              {genres.map((genre, index) => (
                <div
                  key={genre}
                  className="flex items-center gap-2 rounded-xl bg-[#F4F1EE] px-3 py-3"
                >
                  <span className="font-mono text-[10px] text-[#C96F70]">
                    {String(index + 1).padStart(2, "0")}
                  </span>

                  <span className="text-sm font-medium">{genre}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-black/10 bg-white p-7">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
              <Settings2 size={14} />
              Production configuration
            </div>

            <div className="mt-6 divide-y divide-black/5">
              {trainingDetails.map(([label, value]) => (
                <div
                  key={label}
                  className="flex items-center justify-between gap-5 py-3"
                >
                  <span className="text-sm text-[#77716E]">{label}</span>

                  <span className="text-right font-mono text-xs font-medium">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pipeline */}
        <section className="mt-8 rounded-3xl border border-black/10 bg-white p-7 lg:p-10">
          <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
            End-to-end pipeline
          </div>

          <div className="mt-7 grid gap-3 md:grid-cols-5">
            {[
              ["01", "Audio", "Input track"],
              ["02", "Segments", "3-second windows"],
              ["03", "Mel", "128-band representation"],
              ["04", "CNN", "Feature learning"],
              ["05", "Genre", "20-class prediction"],
            ].map(([number, title, detail]) => (
              <div
                key={number}
                className="rounded-2xl bg-[#F4F1EE] p-5"
              >
                <div className="font-mono text-xs text-[#C96F70]">
                  {number}
                </div>

                <div className="mt-4 font-semibold">{title}</div>

                <div className="mt-1 text-xs leading-5 text-[#77716E]">
                  {detail}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Architecture comparison */}
        <section className="mt-8 rounded-3xl border border-black/10 bg-white p-7 lg:p-10">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
            <Cpu size={14} />
            Architecture comparison
          </div>

          <p className="mt-3 max-w-3xl text-sm leading-6 text-[#77716E]">
            Two candidate architectures were evaluated using the same
            expanded dataset and evaluation protocol before the production
            checkpoint was frozen.
          </p>

          <div className="mt-7 overflow-x-auto rounded-2xl border border-black/5">
            <div className="grid min-w-[720px] grid-cols-[1.3fr_1fr_1fr_1fr_0.8fr] bg-[#161616] px-5 py-4 font-mono text-[10px] uppercase tracking-wider text-white/55">
              <span>Architecture</span>
              <span>Parameters</span>
              <span>Track accuracy</span>
              <span>Track Macro F1</span>
              <span>Status</span>
            </div>

            {architectureComparison.map((model) => (
              <div
                key={model.name}
                className="grid min-w-[720px] grid-cols-[1.3fr_1fr_1fr_1fr_0.8fr] items-center border-t border-black/5 px-5 py-4 text-sm"
              >
                <span className="font-medium">{model.name}</span>

                <span className="font-mono text-xs text-[#77716E]">
                  {model.params}
                </span>

                <span className="font-mono text-xs">
                  {model.accuracy}
                </span>

                <span className="font-mono text-xs">
                  {model.f1}
                </span>

                <span
                  className={
                    model.status === "Production"
                      ? "font-mono text-[10px] uppercase tracking-wider text-[#C96F70]"
                      : "font-mono text-[10px] uppercase tracking-wider text-[#77716E]"
                  }
                >
                  {model.status}
                </span>
              </div>
            ))}
          </div>
        </section>

        {/* Verification */}
        <section className="mt-8 rounded-3xl bg-[#C96F70] p-7 text-white lg:p-9">
          <div className="flex items-start gap-4">
            <CheckCircle2 className="mt-1 shrink-0" size={22} />

            <div>
              <div className="font-mono text-xs uppercase tracking-[0.2em] text-white/65">
                Production status
              </div>

              <h2 className="mt-2 text-2xl font-bold">
                Frozen, loaded and verified
              </h2>

              <p className="mt-3 max-w-3xl text-sm leading-6 text-white/75">
                The production checkpoint is the 20-class CustomCNN model.
                The backend loads the frozen checkpoint and has been verified
                through direct inference, spectrogram generation, Grad-CAM
                generation, and the FastAPI prediction pipeline.
              </p>
            </div>
          </div>
        </section>
      </section>

      <footer className="border-t border-black/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-8 font-mono text-xs text-[#77716E] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span>GENGROGRAM / MODEL</span>
          <span>CustomCNN · 393,940 PARAMETERS · 20 CLASSES</span>
        </div>
      </footer>
    </main>
  );
}