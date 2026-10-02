"use client";
import Navbar from "../../components/Navbar";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Clock3,
  Trash2,
  Music2,
  ArrowUpRight,
} from "lucide-react";
import { motion } from "framer-motion";

type HistoryItem = {
  id: string;
  filename: string;
  genre: string;
  confidence: number;
  timestamp: string;
};

const STORAGE_KEY = "genrogram_history";

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);

      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch {
      setHistory([]);
    }
  }, []);

  const clearHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  };

  const formatGenre = (genre: string) =>
    genre.charAt(0).toUpperCase() + genre.slice(1);

  return (
    <main className="min-h-screen bg-[#F4F1EE] text-[#161616]">
      <Navbar />

      <section className="mx-auto max-w-5xl px-6 pb-20 pt-14 lg:px-10 lg:pt-20">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 py-2 font-mono text-xs uppercase tracking-wider text-[#77716E]">
            <Clock3 size={13} />
            Classification History
          </div>

          <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
            <div>
              <h1 className="text-5xl font-bold leading-[0.98] tracking-[-0.045em] sm:text-6xl">
                Your recent
                <span className="block text-[#C96F70]">
                  classifications.
                </span>
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-8 text-[#77716E]">
                Previously classified tracks and their predicted
                genres.
              </p>
            </div>

            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="flex items-center justify-center gap-2 rounded-xl border border-black/10 bg-white px-4 py-3 font-mono text-xs uppercase tracking-wider text-[#77716E] transition hover:border-red-200 hover:text-red-600"
              >
                <Trash2 size={14} />
                Clear history
              </button>
            )}
          </div>
        </motion.div>

        {history.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mt-12 rounded-3xl border border-black/10 bg-white p-12 text-center"
          >
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-[#F4F1EE]">
              <Clock3 size={25} className="text-[#C96F70]" />
            </div>

            <h2 className="mt-6 text-2xl font-bold">
              No classifications yet
            </h2>

            <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-[#77716E]">
              Classify an audio track and your recent result will
              appear here.
            </p>

            <Link
              href="/"
              className="mx-auto mt-7 inline-flex items-center gap-2 rounded-xl bg-[#161616] px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-[#C96F70]"
            >
              Classify a track
              <ArrowUpRight size={16} />
            </Link>
          </motion.div>
        ) : (
          <div className="mt-10 space-y-3">
            {history.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.04 }}
                className="flex flex-col gap-5 rounded-3xl border border-black/10 bg-white p-5 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex min-w-0 items-center gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[#C96F70] text-white">
                    <Music2 size={19} />
                  </div>

                  <div className="min-w-0">
                    <div className="truncate font-semibold">
                      {item.filename}
                    </div>

                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs text-[#77716E]">
                      <span>
                        {new Date(item.timestamp).toLocaleString()}
                      </span>

                      <span>
                        {item.confidence.toFixed(2)}% confidence
                      </span>
                    </div>
                  </div>
                </div>

                <div className="shrink-0 rounded-2xl bg-[#F4F1EE] px-5 py-3">
                  <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-[#77716E]">
                    Predicted genre
                  </div>

                  <div className="mt-1 text-lg font-bold">
                    {formatGenre(item.genre)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        <div className="mt-8 rounded-2xl border border-black/10 bg-white/60 p-5 text-sm leading-6 text-[#77716E]">
          <strong className="text-[#161616]">
            Storage note:
          </strong>{" "}
          History is currently stored locally in this browser.
        </div>
      </section>
    </main>
  );
}
