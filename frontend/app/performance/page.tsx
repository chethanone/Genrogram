"use client";

import Navbar from "../../components/Navbar";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ClipboardCheck,
  Database,
  GitBranch,
  Target,
} from "lucide-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "/api").replace(/\/$/, "");

const trackMetrics = [
  {
    label: "Accuracy",
    value: "75.54%",
    detail: "Correct track-level predictions",
  },
  {
    label: "Macro Precision",
    value: "78.72%",
    detail: "Average precision across 20 genres",
  },
  {
    label: "Macro Recall",
    value: "75.68%",
    detail: "Average recall across 20 genres",
  },
  {
    label: "Macro F1",
    value: "75.58%",
    detail: "Balanced precision and recall",
  },
];

const segmentMetrics = [
  ["Accuracy", "65.56%"],
  ["Macro Precision", "68.30%"],
  ["Macro Recall", "65.47%"],
  ["Macro F1", "65.55%"],
];

const evaluationSteps = [
  {
    number: "01",
    title: "Track-level split",
    description:
      "Original tracks were separated into training, validation, and held-out test sets before segment extraction.",
  },
  {
    number: "02",
    title: "Segment extraction",
    description:
      "Each track is represented using non-overlapping 3-second audio segments.",
  },
  {
    number: "03",
    title: "Segment prediction",
    description:
      "The CustomCNN produces a probability distribution for every extracted segment.",
  },
  {
    number: "04",
    title: "Track aggregation",
    description:
      "Segment probability vectors are averaged to obtain the final track-level prediction.",
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

export default function PerformancePage() {
  const [matrixLoaded, setMatrixLoaded] = useState(false);
  const [matrixExpanded, setMatrixExpanded] = useState(false);
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 767px)");
    const update = () => setIsSmallScreen(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!isSmallScreen || !matrixExpanded) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isSmallScreen, matrixExpanded]);

  useEffect(() => {
    if (!isSmallScreen) setMatrixExpanded(false);
  }, [isSmallScreen]);

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
            <Activity size={13} />
            Model Performance
          </div>

          <h1 className="text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl">
            Measure the
            <span className="block text-[#C96F70]">classifier.</span>
          </h1>

          <p className="mt-6 max-w-2xl text-lg leading-8 text-[#77716E]">
            GENGROGRAM is evaluated on an untouched held-out test set at both
            segment and original-track levels. Track-level results aggregate
            the predictions from multiple audio segments.
          </p>
        </motion.div>

        {/* Evaluation banner */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="mt-10 flex flex-col gap-4 rounded-3xl bg-[#161616] p-6 text-white sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#C96F70]">
              <ClipboardCheck size={21} />
            </div>

            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-white/45">
                Final evaluation
              </div>

              <div className="mt-1 text-lg font-semibold">
                Untouched held-out test set
              </div>
            </div>
          </div>

          <div className="font-mono text-xs text-white/50">
            323 TEST TRACKS · 20 GENRES
          </div>
        </motion.div>

        {/* Track metrics */}
        <section className="mt-8">
          <div className="mb-5">
            <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
              Track-level evaluation
            </div>

            <h2 className="mt-2 text-2xl font-bold">
              Final test performance
            </h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {trackMetrics.map((metric, index) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.06 }}
                className={`rounded-3xl p-5 ${index === 0
                  ? "bg-[#C96F70] text-white"
                  : "border border-black/10 bg-white"
                  }`}
              >
                <div
                  className={`font-mono text-[10px] uppercase tracking-[0.18em] ${index === 0 ? "text-white/65" : "text-[#77716E]"
                    }`}
                >
                  {metric.label}
                </div>

                <div className="mt-4 text-3xl font-bold tracking-tight">
                  {metric.value}
                </div>

                <div
                  className={`mt-3 text-xs leading-5 ${index === 0 ? "text-white/65" : "text-[#77716E]"
                    }`}
                >
                  {metric.detail}
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Segment vs track */}
        <section className="mt-10 grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-black/10 bg-white p-7">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
              <GitBranch size={14} />
              Two evaluation levels
            </div>

            <div className="mt-7 overflow-x-auto rounded-2xl border border-black/5">
              <div className="grid min-w-[420px] grid-cols-3 bg-[#161616] px-5 py-4 font-mono text-[10px] uppercase tracking-wider text-white/55">
                <span>Metric</span>
                <span>Segment</span>
                <span>Track</span>
              </div>

              {[
                ["Accuracy", "65.56%", "75.54%"],
                ["Macro Precision", "68.30%", "78.72%"],
                ["Macro Recall", "65.47%", "75.68%"],
                ["Macro F1", "65.55%", "75.58%"],
              ].map(([metric, segment, track]) => (
                <div
                  key={metric}
                  className="grid min-w-[420px] grid-cols-3 border-t border-black/5 px-5 py-4 text-sm"
                >
                  <span className="font-medium">{metric}</span>

                  <span className="font-mono text-xs text-[#77716E]">
                    {segment}
                  </span>

                  <span className="font-mono text-xs font-medium">
                    {track}
                  </span>
                </div>
              ))}
            </div>

            <p className="mt-5 text-sm leading-6 text-[#77716E]">
              Segment metrics evaluate individual 3-second windows, while
              track-level metrics combine the probability vectors from all
              segments belonging to the same original recording.
            </p>
          </div>

          {/* Test set */}
          <div className="rounded-3xl border border-black/10 bg-white p-7">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
              <Database size={14} />
              Held-out test set
            </div>

            <div className="mt-7">
              <div className="text-5xl font-bold tracking-tight">323</div>

              <div className="mt-1 text-sm text-[#77716E]">
                original tracks
              </div>
            </div>

            <div className="mt-7 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {genres.map((genre, index) => (
                <div
                  key={genre}
                  className="flex items-center gap-2 rounded-xl bg-[#F4F1EE] px-3 py-2.5 text-xs"
                >
                  <span className="font-mono text-[9px] text-[#C96F70]">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span>{genre}</span>
                </div>
              ))}
            </div>
          </div>
        </section>


        {/* Confusion matrix */}
        <section className="mt-8 rounded-3xl border border-black/10 bg-white p-7 lg:p-8">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.2em] text-[#77716E]">
                Class-wise results
              </div>

              <h2 className="mt-2 text-2xl font-bold">
                Confusion matrix
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-[#77716E]">
                The matrix shows how the final track-level predictions are
                distributed across the 20 genre classes.
              </p>
            </div>

            <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-[#AAA4A0]">
              <Target size={13} />
              Held-out test set
            </div>
          </div>

          <div className="mt-7 overflow-hidden rounded-2xl bg-[#161616] p-3 sm:p-6">
            <img
              src={`${API_URL}/evaluation/confusion-matrix`}
              alt="GENGROGRAM 20-class track-level confusion matrix"
              className={`mx-auto block w-full object-contain transition-opacity duration-200 sm:max-h-[700px] ${matrixLoaded ? "opacity-100" : "opacity-0"}`}
              loading="eager"
              decoding="async"
              onLoad={() => setMatrixLoaded(true)}
            />
          </div>

          {isSmallScreen && matrixExpanded && (
            <div
              className="fixed inset-0 z-[100] flex items-center justify-center bg-[#161616]/95 p-3"
              role="dialog"
              aria-modal="true"
              aria-label="Expanded confusion matrix"
              onClick={() => setMatrixExpanded(false)}
            >
              <button
                type="button"
                className="absolute right-4 top-4 z-10 flex h-10 w-10 items-center justify-center rounded-full bg-white text-xl font-medium text-[#161616] shadow-lg"
                onClick={() => setMatrixExpanded(false)}
                aria-label="Close expanded confusion matrix"
              >
                ×
              </button>
              <img
                src={`${API_URL}/evaluation/confusion-matrix`}
                alt="Expanded GENGROGRAM 20-class track-level confusion matrix"
                className="max-h-[94vh] max-w-[96vw] object-contain"
                onClick={(event) => event.stopPropagation()}
              />
            </div>
          )}
        </section>

        {/* Methodology */}
        <section className="mt-8 rounded-3xl bg-[#161616] p-7 text-white lg:p-10">
          <div className="font-mono text-xs uppercase tracking-[0.2em] text-white/45">
            Evaluation methodology
          </div>

          <h2 className="mt-3 text-2xl font-bold">
            From tracks to final predictions
          </h2>

          <div className="mt-8 grid gap-3 md:grid-cols-2">
            {evaluationSteps.map((step) => (
              <div
                key={step.number}
                className="rounded-2xl border border-white/10 bg-white/[0.04] p-5"
              >
                <div className="font-mono text-xs text-[#C96F70]">
                  {step.number}
                </div>

                <h3 className="mt-3 text-lg font-semibold">
                  {step.title}
                </h3>

                <p className="mt-2 text-sm leading-6 text-white/50">
                  {step.description}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-2xl border border-[#C96F70]/30 bg-[#C96F70]/10 p-5">
            <div className="flex items-start gap-3">
              <CheckCircle2
                size={18}
                className="mt-0.5 shrink-0 text-[#E5A0A1]"
              />

              <p className="text-sm leading-6 text-white/70">
                The test tracks remain separate from model training and
                validation. Final track-level metrics are therefore calculated
                from predictions on previously unseen original recordings.
              </p>
            </div>
          </div>
        </section>
      </section>

      <footer className="border-t border-black/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-8 font-mono text-xs text-[#77716E] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span>GENGROGRAM / PERFORMANCE</span>
          <span>HELD-OUT TEST · 323 TRACKS · 20 CLASSES</span>
        </div>
      </footer>
    </main>
  );
}