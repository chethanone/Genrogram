"use client";

import { createContext, useContext, useMemo, useRef, useState } from "react";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

export type PredictionItem = { genre: string; probability: number };
export type PredictionResult = {
  genre: string;
  confidence: number;
  probabilities: Record<string, number>;
  top_predictions: PredictionItem[];
  segment_count: number;
};

type AudioAnalysisContextValue = {
  file: File | null;
  audioUrl: string | null;
  result: PredictionResult | null;
  spectrogramUrl: string | null;
  gradcamUrl: string | null;
  onsetSeconds: number | null;
  gradcamSegmentIndex: number | null;
  processing: boolean;
  error: string | null;
  selectFile: (file: File) => void;
  processTrack: () => Promise<void>;
  clearTrack: () => void;
};

const AudioAnalysisContext = createContext<AudioAnalysisContextValue | null>(null);

export function AudioAnalysisProvider({ children }: { children: React.ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [spectrogramUrl, setSpectrogramUrl] = useState<string | null>(null);
  const [gradcamUrl, setGradcamUrl] = useState<string | null>(null);
  const [onsetSeconds, setOnsetSeconds] = useState<number | null>(null);
  const [gradcamSegmentIndex, setGradcamSegmentIndex] = useState<number | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  const revoke = (url: string | null) => {
    if (url) URL.revokeObjectURL(url);
  };

  const selectFile = (selectedFile: File) => {
    const extension = selectedFile.name.split(".").pop()?.toLowerCase();
    const allowed = ["wav", "mp3", "flac", "ogg", "m4a"];
    if (!extension || !allowed.includes(extension)) {
      setError("Please upload a WAV, MP3, FLAC, OGG, or M4A audio file.");
      return;
    }

    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setError("Please choose an audio file smaller than 4 MB for the web demo.");
      return;
    }

    requestId.current += 1;
    revoke(audioUrl);
    revoke(spectrogramUrl);
    revoke(gradcamUrl);

    setFile(selectedFile);
    setAudioUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setSpectrogramUrl(null);
    setGradcamUrl(null);
    setOnsetSeconds(null);
    setGradcamSegmentIndex(null);
    setError(null);
  };

  const processTrack = async () => {
    if (!file || processing) return;
    const id = ++requestId.current;
    setProcessing(true);
    setError(null);

    try {
      const predictionForm = new FormData();
      predictionForm.append("file", file);
      const predictionResponse = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: predictionForm,
      });
      if (!predictionResponse.ok) throw new Error(await predictionResponse.text() || "Classification failed.");
      const prediction: PredictionResult = await predictionResponse.json();
      if (id !== requestId.current) return;
      setResult(prediction);

      const existingHistory = JSON.parse(localStorage.getItem("genrogram_history") || "[]");
      localStorage.setItem("genrogram_history", JSON.stringify([
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          filename: file.name,
          genre: prediction.genre,
          confidence: prediction.confidence * 100,
          timestamp: new Date().toISOString(),
        },
        ...existingHistory,
      ].slice(0, 50)));

      const spectrogramForm = new FormData();
      spectrogramForm.append("file", file);
      const explainForm = new FormData();
      explainForm.append("file", file);
      explainForm.append("target_genre", prediction.genre);

      const [spectrogramResponse, explainResponse] = await Promise.all([
        fetch(`${API_URL}/spectrogram`, { method: "POST", body: spectrogramForm }),
        fetch(`${API_URL}/explainability`, { method: "POST", body: explainForm }),
      ]);

      if (!spectrogramResponse.ok) throw new Error(await spectrogramResponse.text() || "Spectrogram generation failed.");
      if (!explainResponse.ok) throw new Error(await explainResponse.text() || "Explainability generation failed.");

      const spectrogramBlob = await spectrogramResponse.blob();
      const gradcamBlob = await explainResponse.blob();
      const nextSpectrogramUrl = URL.createObjectURL(spectrogramBlob);
      const nextGradcamUrl = URL.createObjectURL(gradcamBlob);

      if (id !== requestId.current) {
        URL.revokeObjectURL(nextSpectrogramUrl);
        URL.revokeObjectURL(nextGradcamUrl);
        return;
      }

      revoke(spectrogramUrl);
      revoke(gradcamUrl);
      setSpectrogramUrl(nextSpectrogramUrl);
      setGradcamUrl(nextGradcamUrl);
      const onsetHeader = spectrogramResponse.headers.get("X-Onset-Seconds");
      const segmentHeader = explainResponse.headers.get("X-Segment-Index");
      setOnsetSeconds(onsetHeader ? Number(onsetHeader) : null);
      setGradcamSegmentIndex(segmentHeader ? Number(segmentHeader) : null);
    } catch (err) {
      if (id === requestId.current) {
        setError(err instanceof Error ? err.message : "Something went wrong while analyzing the track.");
      }
    } finally {
      if (id === requestId.current) setProcessing(false);
    }
  };

  const clearTrack = () => {
    requestId.current += 1;
    revoke(audioUrl);
    revoke(spectrogramUrl);
    revoke(gradcamUrl);
    setFile(null);
    setAudioUrl(null);
    setResult(null);
    setSpectrogramUrl(null);
    setGradcamUrl(null);
    setOnsetSeconds(null);
    setGradcamSegmentIndex(null);
    setProcessing(false);
    setError(null);
  };

  const value = useMemo(() => ({ file, audioUrl, result, spectrogramUrl, gradcamUrl, onsetSeconds, gradcamSegmentIndex, processing, error, selectFile, processTrack, clearTrack }), [file, audioUrl, result, spectrogramUrl, gradcamUrl, onsetSeconds, gradcamSegmentIndex, processing, error]);
  return <AudioAnalysisContext.Provider value={value}>{children}</AudioAnalysisContext.Provider>;
}

export function useAudioAnalysis() {
  const value = useContext(AudioAnalysisContext);
  if (!value) throw new Error("useAudioAnalysis must be used inside AudioAnalysisProvider");
  return value;
}
