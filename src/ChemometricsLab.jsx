import React, { useState, useMemo } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, LineChart, Line, Dot,
  BarChart, Bar, Cell,
} from "recharts";
import {
  TrendingUp, Layers, Waves, Brain, Dna, Upload, FlaskConical,
  Play, Trophy, Loader2, AlertCircle, Beaker, RotateCcw, ChevronRight, ScanLine,
} from "lucide-react";

/* ───────────────────────── Linear algebra helpers ───────────────────────── */

const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const std = (a) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1 || 1)) || 1;
};
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const normalizeVec = (v) => {
  const n = Math.sqrt(dot(v, v));
  return n < 1e-12 ? v.slice() : v.map((x) => x / n);
};
const transpose = (M) => M[0].map((_, c) => M.map((row) => row[c]));
const matMul = (A, B) => {
  const n = A.length, m = A[0].length, p = B[0].length;
  const R = Array.from({ length: n }, () => Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++) R[i][j] += aik * B[k][j];
    }
  }
  return R;
};
const matVec = (M, v) => M.map((row) => dot(row, v));
const ridge = (M, lambda) => M.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));

function invertMatrix(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let pivotRow = col, maxVal = Math.abs(A[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > maxVal) { maxVal = Math.abs(A[r][col]); pivotRow = r; }
    }
    if (maxVal < 1e-12) A[col][col] += 1e-6;
    if (pivotRow !== col) { const tmp = A[col]; A[col] = A[pivotRow]; A[pivotRow] = tmp; }
    const pivot = A[col][col];
    for (let j = 0; j < 2 * n; j++) A[col][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = A[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[r][j] -= factor * A[col][j];
    }
  }
  return A.map((row) => row.slice(n));
}

function standardize(X) {
  const p = X[0].length;
  const means = [], stds = [];
  for (let j = 0; j < p; j++) {
    const col = X.map((row) => row[j]);
    means.push(mean(col));
    stds.push(std(col));
  }
  const Z = X.map((row) => row.map((v, j) => (v - means[j]) / stds[j]));
  return { Z, means, stds };
}
const applyStandardize = (X, means, stds) => X.map((row) => row.map((v, j) => (v - means[j]) / (stds[j] || 1)));

const r2 = (yTrue, yPred) => {
  const m = mean(yTrue);
  const ssTot = yTrue.reduce((s, y) => s + (y - m) * (y - m), 0);
  const ssRes = yTrue.reduce((s, y, i) => s + (y - yPred[i]) * (y - yPred[i]), 0);
  return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
};
const rmseFn = (yTrue, yPred) => Math.sqrt(yTrue.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0) / yTrue.length);

function trainTestSplit(X, y, trainRatio = 0.8) {
  const n = X.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTrain = Math.max(2, Math.round(n * trainRatio));
  const trainIdx = idx.slice(0, nTrain);
  const testIdx = idx.slice(nTrain).length ? idx.slice(nTrain) : idx.slice(0, Math.max(1, Math.floor(n * 0.2)));
  return {
    Xtrain: trainIdx.map((i) => X[i]), ytrain: trainIdx.map((i) => y[i]),
    Xtest: testIdx.map((i) => X[i]), ytest: testIdx.map((i) => y[i]),
  };
}

/* ───────────────────────── Models ───────────────────────── */

function fitMLR(Xtrain, ytrain) {
  const Xb = Xtrain.map((row) => [1, ...row]);
  const Xt = transpose(Xb);
  let XtX = matMul(Xt, Xb);
  XtX = ridge(XtX, 1e-6);
  const XtXinv = invertMatrix(XtX);
  const Xty = matVec(Xt, ytrain);
  return matVec(XtXinv, Xty);
}
const predictMLR = (X, beta) => X.map((row) => beta[0] + dot(row, beta.slice(1)));

function fitPLS(Xtrain, ytrain, ncomp) {
  const { Z: Xz, means: xMeans, stds: xStds } = standardize(Xtrain);
  const yMean = mean(ytrain);
  let yc = ytrain.map((v) => v - yMean);
  let Xres = Xz.map((row) => row.slice());
  const W = [], P = [], Q = [];
  for (let a = 0; a < ncomp; a++) {
    let w = matVec(transpose(Xres), yc);
    w = normalizeVec(w);
    const t = matVec(Xres, w);
    const ttt = dot(t, t) || 1e-12;
    const p_load = matVec(transpose(Xres), t).map((v) => v / ttt);
    const q = dot(yc, t) / ttt;
    W.push(w); P.push(p_load); Q.push(q);
    Xres = Xres.map((row, i) => row.map((v, j) => v - t[i] * p_load[j]));
    yc = yc.map((v, i) => v - t[i] * q);
  }
  const Wm = transpose(W), Pm = transpose(P);
  const PtW = ridge(matMul(transpose(Pm), Wm), 1e-8);
  const Bcomp = matVec(invertMatrix(PtW), Q);
  const B = matVec(Wm, Bcomp);
  return { B, xMeans, xStds, yMean };
}
const predictPLS = (X, m) => applyStandardize(X, m.xMeans, m.xStds).map((row) => m.yMean + dot(row, m.B));

function topEigenvectors(C, k, p) {
  let remaining = C.map((row) => row.slice());
  const vectors = [];
  for (let c = 0; c < k; c++) {
    let v = normalizeVec(Array.from({ length: p }, () => Math.random() - 0.5));
    for (let iter = 0; iter < 200; iter++) {
      const nv = matVec(remaining, v);
      const norm = Math.sqrt(dot(nv, nv));
      if (norm < 1e-10) break;
      v = nv.map((x) => x / norm);
    }
    const eigenvalue = dot(v, matVec(remaining, v));
    vectors.push(v);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) remaining[i][j] -= eigenvalue * v[i] * v[j];
  }
  return vectors;
}

function fitPCR(Xtrain, ytrain, ncomp) {
  const { Z: Xz, means: xMeans, stds: xStds } = standardize(Xtrain);
  const yMean = mean(ytrain);
  const yc = ytrain.map((v) => v - yMean);
  const n = Xz.length, p = Xz[0].length;
  const k = Math.min(ncomp, p, n - 1);
  const C = matMul(transpose(Xz), Xz).map((row) => row.map((v) => v / (n - 1)));
  const vectors = topEigenvectors(C, k, p);
  const V = transpose(vectors);
  const T = matMul(Xz, V);
  const TtT = ridge(matMul(transpose(T), T), 1e-8);
  const Tty = matVec(transpose(T), yc);
  const betaScores = matVec(invertMatrix(TtT), Tty);
  const B = matVec(V, betaScores);
  return { B, xMeans, xStds, yMean, kUsed: k };
}
const predictPCR = (X, m) => applyStandardize(X, m.xMeans, m.xStds).map((row) => m.yMean + dot(row, m.B));

function fitANN(Xtrain, ytrain, { hidden, epochs, lr }) {
  const { Z: Xz, means: xMeans, stds: xStds } = standardize(Xtrain);
  const yStdVal = std(ytrain), yMean = mean(ytrain);
  const yz = ytrain.map((v) => (v - yMean) / yStdVal);
  const n = Xz.length, p = Xz[0].length;
  let W1 = Array.from({ length: p }, () => Array.from({ length: hidden }, () => (Math.random() * 2 - 1) * Math.sqrt(1 / p)));
  let b1 = Array(hidden).fill(0);
  let W2 = Array.from({ length: hidden }, () => (Math.random() * 2 - 1) * Math.sqrt(1 / hidden));
  let b2 = 0;
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  for (let epoch = 0; epoch < epochs; epoch++) {
    const A1 = Xz.map((row) => Array.from({ length: hidden }, (_, h) => sigmoid(b1[h] + row.reduce((s, x, i) => s + x * W1[i][h], 0))));
    const yhat = A1.map((row) => b2 + row.reduce((s, a, h) => s + a * W2[h], 0));
    const dZ2 = yhat.map((yh, i) => (2 / n) * (yh - yz[i]));
    const dW2 = Array(hidden).fill(0);
    for (let h = 0; h < hidden; h++) { let s = 0; for (let i = 0; i < n; i++) s += dZ2[i] * A1[i][h]; dW2[h] = s; }
    const db2 = dZ2.reduce((s, v) => s + v, 0);
    const dW1 = Array.from({ length: p }, () => Array(hidden).fill(0));
    const db1 = Array(hidden).fill(0);
    for (let i = 0; i < n; i++) {
      for (let h = 0; h < hidden; h++) {
        const a = A1[i][h];
        const dZ1ih = dZ2[i] * W2[h] * a * (1 - a);
        db1[h] += dZ1ih;
        for (let j = 0; j < p; j++) dW1[j][h] += dZ1ih * Xz[i][j];
      }
    }
    for (let h = 0; h < hidden; h++) {
      W2[h] -= lr * dW2[h];
      b1[h] -= lr * db1[h];
      for (let j = 0; j < p; j++) W1[j][h] -= lr * dW1[j][h];
    }
    b2 -= lr * db2;
  }
  return { W1, b1, W2, b2, xMeans, xStds, yMean, yStd: yStdVal };
}
function predictANNFixed(X, m) {
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  return applyStandardize(X, m.xMeans, m.xStds).map((row) => {
    const A1 = Array.from({ length: m.b1.length }, (_, h) => sigmoid(m.b1[h] + row.reduce((s, x, i) => s + x * m.W1[i][h], 0)));
    const yhatZ = m.b2 + A1.reduce((s, a, h) => s + a * m.W2[h], 0);
    return yhatZ * m.yStd + m.yMean;
  });
}

function tournamentSelect(pop, scores, k = 3) {
  let bestIdx = 0, bestScore = -Infinity;
  for (let i = 0; i < k; i++) {
    const idx = Math.floor(Math.random() * pop.length);
    if (scores[idx] > bestScore) { bestScore = scores[idx]; bestIdx = idx; }
  }
  return pop[bestIdx];
}

function fitGA(Xtrain, ytrain, Xtest, ytest, { popSize, generations, mutationRate, maxVars }) {
  const p = Xtrain[0].length;
  const maxV = Math.min(maxVars, p);
  const selectCols = (X, chrom) => {
    const idxs = chrom.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    return { idxs, Xs: X.map((row) => idxs.map((i) => row[i])) };
  };
  const capChrom = (chrom) => {
    let onesIdx = chrom.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    if (onesIdx.length < 2) {
      const zerosIdx = chrom.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0).sort(() => Math.random() - 0.5);
      zerosIdx.slice(0, 2 - onesIdx.length).forEach((i) => (chrom[i] = 1));
      onesIdx = chrom.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    }
    if (onesIdx.length > maxV) {
      onesIdx.sort(() => Math.random() - 0.5).slice(maxV).forEach((i) => (chrom[i] = 0));
    }
    return chrom;
  };
  const randomChromosome = () => capChrom(Array.from({ length: p }, () => (Math.random() < 0.3 ? 1 : 0)));
  const fitness = (chrom) => {
    const { Xs: XsTrain } = selectCols(Xtrain, chrom);
    const { Xs: XsTest } = selectCols(Xtest, chrom);
    if (XsTrain[0].length === 0) return -Infinity;
    try {
      const beta = fitMLR(XsTrain, ytrain);
      return r2(ytest, predictMLR(XsTest, beta));
    } catch { return -Infinity; }
  };
  let population = Array.from({ length: popSize }, randomChromosome);
  let scores = population.map(fitness);
  let best = population[0].slice(), bestScore = -Infinity;
  for (let gen = 0; gen < generations; gen++) {
    for (let i = 0; i < population.length; i++) if (scores[i] > bestScore) { bestScore = scores[i]; best = population[i].slice(); }
    const newPop = [best.slice()];
    while (newPop.length < popSize) {
      const a = tournamentSelect(population, scores), b = tournamentSelect(population, scores);
      let child = a.map((g, i) => (Math.random() < 0.5 ? g : b[i]));
      child = child.map((g) => (Math.random() < mutationRate ? 1 - g : g));
      newPop.push(capChrom(child));
    }
    population = newPop;
    scores = population.map(fitness);
  }
  for (let i = 0; i < population.length; i++) if (scores[i] > bestScore) { bestScore = scores[i]; best = population[i].slice(); }
  const { idxs, Xs: XsTrainFinal } = selectCols(Xtrain, best);
  const beta = fitMLR(XsTrainFinal, ytrain);
  return { chromosome: best, idxs, beta };
}
const predictGA = (X, m) => predictMLR(X.map((row) => m.idxs.map((i) => row[i])), m.beta);

function kFoldIndices(n, k) {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const folds = Array.from({ length: k }, () => []);
  idx.forEach((v, i) => folds[i % k].push(v));
  return folds;
}

// RMSECV vs. number of latent variables, via leave-one-out cross-validation (for PLS / PCR)
function rmsecvCurve(X, y, methodId, maxComp, k = X.length) {
  const n = X.length;
  const kUse = Math.max(2, Math.min(k, n));
  const folds = kFoldIndices(n, kUse);
  const curve = [];
  for (let ncomp = 1; ncomp <= maxComp; ncomp++) {
    let sqErrSum = 0, count = 0;
    for (let f = 0; f < kUse; f++) {
      const testIdx = folds[f];
      const trainIdx = folds.flatMap((fold, fi) => (fi === f ? [] : fold));
      if (trainIdx.length < ncomp + 2 || testIdx.length === 0) continue;
      const Xtr = trainIdx.map((i) => X[i]), ytr = trainIdx.map((i) => y[i]);
      const Xte = testIdx.map((i) => X[i]), yte = testIdx.map((i) => y[i]);
      try {
        let model, pred;
        if (methodId === "pls") { model = fitPLS(Xtr, ytr, ncomp); pred = predictPLS(Xte, model); }
        else { model = fitPCR(Xtr, ytr, ncomp); pred = predictPCR(Xte, model); }
        yte.forEach((v, i) => { sqErrSum += (v - pred[i]) ** 2; count++; });
      } catch { /* skip unstable fold/ncomp combo */ }
    }
    if (count > 0) curve.push({ ncomp, rmsecv: Math.sqrt(sqErrSum / count) });
  }
  return curve;
}

// Find the optimal number of PLS latent variables by leave-one-out CV (minimum RMSECV)
function optimalPLS_LOOCV(X, y, maxComp) {
  const n = X.length;
  const maxC = Math.max(1, Math.min(maxComp, X[0].length, n - 2));
  let best = null;
  for (let ncomp = 1; ncomp <= maxC; ncomp++) {
    let sqErr = 0, count = 0;
    for (let i = 0; i < n; i++) {
      const Xtr = X.filter((_, idx) => idx !== i), ytr = y.filter((_, idx) => idx !== i);
      try {
        const model = fitPLS(Xtr, ytr, ncomp);
        const pred = predictPLS([X[i]], model);
        sqErr += (y[i] - pred[0]) ** 2; count++;
      } catch { /* skip */ }
    }
    if (count === 0) continue;
    const rmsecv = Math.sqrt(sqErr / count);
    if (!best || rmsecv < best.rmsecv) best = { ncomp, rmsecv };
  }
  if (!best) best = { ncomp: 1, rmsecv: NaN };
  const fullModel = fitPLS(X, y, best.ncomp);
  const r2Val = r2(y, predictPLS(X, fullModel));
  return { ncomp: best.ncomp, rmsecv: best.rmsecv, r2: r2Val };
}

// Pull a numeric wavelength out of a column header like "WL210", "210", "A_210nm"
function extractWavelength(colName) {
  const match = String(colName).match(/(\d+(\.\d+)?)/);
  return match ? parseFloat(match[0]) : NaN;
}
function getWavelengths(xCols) {
  const parsed = xCols.map(extractWavelength);
  return parsed.every(Number.isFinite) ? parsed : xCols.map((_, i) => i);
}

// Evaluate contiguous wavelength intervals, exclude noisy/uninformative regions,
// and recommend the range with the lowest LOOCV RMSECV (interval-PLS style search).
function evaluateRangeSelection(X, y, wavelengths, { nIntervals, maxComp }) {
  const p = X[0].length;
  const order = wavelengths.map((_, i) => i).sort((a, b) => wavelengths[a] - wavelengths[b]);
  const sortedW = order.map((i) => wavelengths[i]);
  const sortedX = X.map((row) => order.map((i) => row[i]));

  const baseline = optimalPLS_LOOCV(sortedX, y, maxComp);

  const intervals = [];
  const chunk = Math.max(2, Math.ceil(p / nIntervals));
  for (let start = 0; start < p; start += chunk) {
    const end = Math.min(start + chunk, p);
    if (end - start < 2) continue;
    const idxs = Array.from({ length: end - start }, (_, k) => start + k);
    const subX = sortedX.map((row) => idxs.map((i) => row[i]));
    const res = optimalPLS_LOOCV(subX, y, Math.min(maxComp, subX[0].length, y.length - 3));
    intervals.push({
      label: `${fmt(sortedW[start], 0)}–${fmt(sortedW[end - 1], 0)}`,
      wMin: sortedW[start], wMax: sortedW[end - 1],
      nVars: idxs.length, rmsecv: res.rmsecv, ncomp: res.ncomp, r2: res.r2,
    });
  }
  const ranked = [...intervals].sort((a, b) => a.rmsecv - b.rmsecv);
  return { baseline, intervals, best: ranked[0], ranked };
}


function runMethod(methodId, X, y, params) {
  const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(X, y, 0.8);
  let predTrain, predTest, model;
  if (methodId === "mlr") {
    const beta = fitMLR(Xtrain, ytrain);
    predTrain = predictMLR(Xtrain, beta); predTest = predictMLR(Xtest, beta); model = { beta };
  } else if (methodId === "pls") {
    const ncomp = Math.max(1, Math.min(params.ncomp, Xtrain.length - 1, Xtrain[0].length));
    model = fitPLS(Xtrain, ytrain, ncomp);
    predTrain = predictPLS(Xtrain, model); predTest = predictPLS(Xtest, model);
  } else if (methodId === "pcr") {
    const ncomp = Math.max(1, Math.min(params.ncomp, Xtrain.length - 1, Xtrain[0].length));
    model = fitPCR(Xtrain, ytrain, ncomp);
    predTrain = predictPCR(Xtrain, model); predTest = predictPCR(Xtest, model);
  } else if (methodId === "ann") {
    model = fitANN(Xtrain, ytrain, { hidden: params.hidden, epochs: params.epochs, lr: params.lr });
    predTrain = predictANNFixed(Xtrain, model); predTest = predictANNFixed(Xtest, model);
  } else if (methodId === "ga") {
    model = fitGA(Xtrain, ytrain, Xtest, ytest, { popSize: params.popSize, generations: params.generations, mutationRate: params.mutationRate, maxVars: params.maxVars });
    predTrain = predictGA(Xtrain, model); predTest = predictGA(Xtest, model);
  }
  const chartData = ytest.map((v, i) => ({ actual: v, predicted: predTest[i] }));
  return {
    r2Train: r2(ytrain, predTrain), r2Test: r2(ytest, predTest),
    rmseTrain: rmseFn(ytrain, predTrain), rmseTest: rmseFn(ytest, predTest),
    chartData, model, nTrain: Xtrain.length, nTest: Xtest.length,
  };
}

/* ───────────────────────── Demo data ───────────────────────── */

function generateDemoData() {
  const n = 40, p = 15;
  const trueCoeffs = Array.from({ length: p }, () => Math.random() * 0.8 + 0.1);
  const baseline = Array.from({ length: p }, () => Math.random() * 0.3);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const conc = 1 + Math.random() * 9;
    const row = {};
    for (let j = 0; j < p; j++) {
      const noise = (Math.random() - 0.5) * 0.05;
      row[`WL${210 + j * 10}`] = +(baseline[j] + trueCoeffs[j] * conc + noise).toFixed(4);
    }
    row["Conc"] = +conc.toFixed(3);
    rows.push(row);
  }
  return { headers: Object.keys(rows[0]), rows };
}

/* ───────────────────────── Constants ───────────────────────── */

const METHODS = [
  { id: "mlr", name: "MLR", full: "Multiple Linear Regression", icon: TrendingUp, accent: "text-sky-400", ring: "ring-sky-500/40", bar: "bg-sky-500" },
  { id: "pcr", name: "PCR", full: "Principal Component Regression", icon: Layers, accent: "text-teal-400", ring: "ring-teal-500/40", bar: "bg-teal-500" },
  { id: "pls", name: "PLS", full: "Partial Least Squares", icon: Waves, accent: "text-emerald-400", ring: "ring-emerald-500/40", bar: "bg-emerald-500" },
  { id: "ann", name: "ANN", full: "Artificial Neural Network", icon: Brain, accent: "text-fuchsia-400", ring: "ring-fuchsia-500/40", bar: "bg-fuchsia-500" },
  { id: "ga", name: "GA-MLR", full: "Genetic Algorithm Variable Selection", icon: Dna, accent: "text-violet-400", ring: "ring-violet-500/40", bar: "bg-violet-500" },
  { id: "range", name: "Range Select", full: "Wavelength Range Selection (iPLS)", icon: ScanLine, accent: "text-orange-400", ring: "ring-orange-500/40", bar: "bg-orange-500" },
];

const OTHER_METHODS = [
  "GA-PLS", "SVR", "CLS", "ILS", "SIMCA", "LDA", "QDA", "KNN",
  "PSO", "UVE", "SNV / MSC", "Savitzky–Golay", "Wavelet Transform", "HCA", "OPLS",
];

function defaultParams(id) {
  if (id === "pls" || id === "pcr") return { ncomp: 3 };
  if (id === "ann") return { hidden: 5, epochs: 300, lr: 0.05 };
  if (id === "ga") return { popSize: 30, generations: 40, mutationRate: 0.05, maxVars: 8 };
  if (id === "range") return { nIntervals: 10, maxComp: 8 };
  return {};
}
function defaultPanel(id) {
  return { headers: [], rows: [], xCols: [], yCol: null, status: "empty", error: null, result: null, params: defaultParams(id), fileName: null, rangeInput: "" };
}

const fmt = (n, d = 4) => (Number.isFinite(n) ? n.toFixed(d) : "—");

/* ───────────────────────── Component ───────────────────────── */

export default function ChemometricsLab() {
  const [activeMethod, setActiveMethod] = useState("pls");
  const [panels, setPanels] = useState(() => Object.fromEntries(METHODS.map((m) => [m.id, defaultPanel(m.id)])));

  const updatePanel = (id, patch) => setPanels((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));

  const bestOverall = useMemo(() => {
    let best = null;
    for (const m of METHODS) {
      if (m.id === "range") continue;
      const res = panels[m.id].result;
      if (res && (!best || res.r2Test > best.r2Test)) best = { ...res, methodId: m.id };
    }
    return best;
  }, [panels]);

  function ingestData(id, headers, rows) {
    const numericCols = headers.filter((h) => rows.some((r) => Number.isFinite(Number(r[h]))));
    const yCol = numericCols[numericCols.length - 1] || null;
    const xCols = numericCols.filter((c) => c !== yCol);
    updatePanel(id, { headers, rows, xCols, yCol, status: "ready", result: null, error: null });
  }

  function handleFile(id, file) {
    const ext = file.name.split(".").pop().toLowerCase();
    updatePanel(id, { fileName: file.name });
    if (ext === "csv") {
      Papa.parse(file, {
        header: true, dynamicTyping: true, skipEmptyLines: true,
        complete: (results) => ingestData(id, results.meta.fields || [], results.data),
        error: (err) => updatePanel(id, { status: "error", error: String(err) }),
      });
    } else {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const wb = XLSX.read(e.target.result, { type: "array" });
          const ws = wb.Sheets[wb.SheetNames[0]];
          const json = XLSX.utils.sheet_to_json(ws, { defval: null });
          const headers = json.length ? Object.keys(json[0]) : [];
          ingestData(id, headers, json);
        } catch (err) { updatePanel(id, { status: "error", error: String(err) }); }
      };
      reader.readAsArrayBuffer(file);
    }
  }

  function handleDemo(id) {
    const { headers, rows } = generateDemoData();
    updatePanel(id, { fileName: "بيانات تجريبية (Demo)" });
    ingestData(id, headers, rows);
  }

  function buildMatrix(panel) {
    const xCols = panel.xCols.filter((c) => c !== panel.yCol);
    const X = [], y = [];
    panel.rows.forEach((row) => {
      const xVals = xCols.map((c) => Number(row[c]));
      const yVal = Number(row[panel.yCol]);
      if (xVals.every(Number.isFinite) && Number.isFinite(yVal)) { X.push(xVals); y.push(yVal); }
    });
    return { X, y, xCols };
  }

  function handleRun(id) {
    const panel = panels[id];
    const { X, y, xCols } = buildMatrix(panel);
    if (X.length < 6 || X[0].length < 1) {
      updatePanel(id, { status: "error", error: "Not enough data — you need at least 6 samples and one independent variable." });
      return;
    }
    updatePanel(id, { status: "running", error: null });
    setTimeout(() => {
      try {
        if (id === "range") {
          if (X[0].length < 4) {
            updatePanel(id, { status: "error", error: "Select at least 4 wavelength columns to search a range within." });
            return;
          }
          const wavelengths = getWavelengths(xCols);
          const rangeRes = evaluateRangeSelection(X, y, wavelengths, panel.params);
          updatePanel(id, { status: "done", result: rangeRes, error: null });
          return;
        }
        const res = runMethod(id, X, y, panel.params);
        if (id === "pls" || id === "pcr") {
          const maxComp = Math.max(panel.params.ncomp, Math.min(15, X[0].length, X.length - 3));
          res.rmsecvCurve = rmsecvCurve(X, y, id, Math.max(1, maxComp)); // leave-one-out CV
        }
        updatePanel(id, { status: "done", result: res, error: null });
      } catch (e) {
        updatePanel(id, { status: "error", error: "Could not build the model: " + (e.message || e) });
      }
    }, 50);
  }

  function toggleX(id, col) {
    const panel = panels[id];
    const has = panel.xCols.includes(col);
    updatePanel(id, { xCols: has ? panel.xCols.filter((c) => c !== col) : [...panel.xCols, col] });
  }

  function applyWavelengthRange(id) {
    const panel = panels[id];
    const nums = (panel.rangeInput.match(/(\d+(\.\d+)?)/g) || []).map(Number);
    if (nums.length < 2) {
      updatePanel(id, { error: 'Enter a range like "230-270" (two numbers).' });
      return;
    }
    const lo = Math.min(nums[0], nums[1]), hi = Math.max(nums[0], nums[1]);
    const candidateCols = panel.headers.filter((h) => h !== panel.yCol);
    const wl = getWavelengths(candidateCols);
    const selected = candidateCols.filter((_, i) => wl[i] >= lo && wl[i] <= hi);
    if (selected.length === 0) {
      updatePanel(id, { error: `No columns found between ${lo} and ${hi}.` });
      return;
    }
    updatePanel(id, { xCols: selected, error: null });
  }

  const activePanel = panels[activeMethod];
  const activeMeta = METHODS.find((m) => m.id === activeMethod);
  const nSelectedX = activePanel.xCols.filter((c) => c !== activePanel.yCol).length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
        .font-display { font-family: 'Space Grotesk', sans-serif; }
        .font-body { font-family: 'Inter', sans-serif; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 8px; }
      `}</style>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 font-body">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-2 text-slate-500 text-xs tracking-widest uppercase mb-2">
            <FlaskConical size={14} />
            <span>Chemometrics Model Builder</span>
          </div>
          <h1 className="font-display text-3xl sm:text-4xl font-bold text-white">Chemometrics Lab</h1>
          <p className="text-slate-400 mt-2 max-w-2xl text-sm sm:text-base">
            Upload your spectral data (CSV or Excel), pick the independent variables (wavelengths) and the
            dependent variable (concentration), build the model with any method, then compare R² to pick the best one.
          </p>
        </div>

        {/* Optical bench rail */}
        <div className="relative h-2 rounded-full mb-8 overflow-hidden"
          style={{ background: "linear-gradient(90deg,#7c3aed,#2563eb,#0d9488,#16a34a,#eab308,#ea580c)" }}>
        </div>

        {/* Best model banner */}
        {bestOverall && (
          <div className="flex items-center gap-3 mb-6 px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300">
            <Trophy size={18} className="shrink-0" />
            <span className="text-sm">
              Best model so far: <b className="font-display">{METHODS.find(m => m.id === bestOverall.methodId)?.name}</b>
              {"  —  R² (test) = "}<span className="font-mono">{fmt(bestOverall.r2Test)}</span>
            </span>
          </div>
        )}

        {/* Method cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
          {METHODS.map((m) => {
            const Icon = m.icon;
            const isActive = activeMethod === m.id;
            const res = panels[m.id].result;
            const isBest = bestOverall && bestOverall.methodId === m.id;
            return (
              <button
                key={m.id}
                onClick={() => setActiveMethod(m.id)}
                className={`relative text-left rounded-xl border px-4 py-4 transition-all bg-slate-900/60 hover:bg-slate-900
                  ${isActive ? `border-slate-600 ring-2 ${m.ring}` : "border-slate-800"}`}
              >
                {isBest && <Trophy size={14} className="absolute top-3 right-3 text-amber-400" />}
                <Icon size={20} className={m.accent} />
                <div className="font-display font-semibold text-white mt-2 text-sm">{m.name}</div>
                <div className="text-[11px] text-slate-500 mt-0.5 leading-tight">{m.full}</div>
                {res && m.id !== "range" && (
                  <div className="mt-2 text-[11px] font-mono text-slate-400">
                    R²={fmt(res.r2Test, 3)}
                  </div>
                )}
                {res && m.id === "range" && (
                  <div className="mt-2 text-[11px] font-mono text-slate-400">
                    best={res.best?.label} nm
                  </div>
                )}
              </button>
            );
          })}
        </div>

        {/* Active panel */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5 sm:p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <activeMeta.icon size={18} className={activeMeta.accent} />
            <h2 className="font-display text-lg font-semibold text-white">{activeMeta.name} — {activeMeta.full}</h2>
          </div>

          {/* Upload row */}
          <div className="flex flex-wrap items-center gap-3 mb-5">
            <label className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 cursor-pointer text-sm text-slate-200 border border-slate-700">
              <Upload size={15} />
              Upload CSV / Excel
              <input type="file" accept=".csv,.xlsx,.xls" className="hidden"
                onChange={(e) => e.target.files[0] && handleFile(activeMethod, e.target.files[0])} />
            </label>
            <button onClick={() => handleDemo(activeMethod)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/60 hover:bg-slate-700 text-sm text-slate-300 border border-slate-800">
              <Beaker size={15} />
              Demo data
            </button>
            {activePanel.fileName && (
              <span className="text-xs text-slate-500 font-mono">{activePanel.fileName}</span>
            )}
            {activePanel.headers.length > 0 && (
              <button onClick={() => updatePanel(activeMethod, defaultPanel(activeMethod))}
                className="ml-auto flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300">
                <RotateCcw size={13} /> Reset
              </button>
            )}
          </div>

          {activePanel.error && (
            <div className="flex items-center gap-2 mb-4 px-4 py-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm">
              <AlertCircle size={15} className="shrink-0" /> {activePanel.error}
            </div>
          )}

          {activePanel.headers.length > 0 && (
            <>
              {/* preview table */}
              <div className="overflow-x-auto mb-5 rounded-lg border border-slate-800">
                <table className="w-full text-xs font-mono">
                  <thead className="bg-slate-800/60 text-slate-400">
                    <tr>{activePanel.headers.map((h) => <th key={h} className="px-3 py-2 text-left whitespace-nowrap">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {activePanel.rows.slice(0, 4).map((row, i) => (
                      <tr key={i} className="border-t border-slate-800/60 text-slate-400">
                        {activePanel.headers.map((h) => <td key={h} className="px-3 py-1.5 whitespace-nowrap">{String(row[h] ?? "")}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* optimum wavelength range selection */}
              <div className="mb-5">
                <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Optimum wavelength selection region</div>
                <div className="flex flex-wrap items-center gap-2">
                  <input type="text" value={activePanel.rangeInput}
                    onChange={(e) => updatePanel(activeMethod, { rangeInput: e.target.value })}
                    onKeyDown={(e) => e.key === "Enter" && applyWavelengthRange(activeMethod)}
                    placeholder="e.g. 230-270"
                    className="w-40 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200 placeholder:text-slate-600" />
                  <button onClick={() => applyWavelengthRange(activeMethod)}
                    className="px-3 py-2 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">
                    Apply range
                  </button>
                  <button onClick={() => {
                      const candidateCols = activePanel.headers.filter((h) => h !== activePanel.yCol);
                      updatePanel(activeMethod, { xCols: candidateCols, rangeInput: "" });
                    }}
                    className="px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300">
                    Use full spectrum
                  </button>
                  <span className="text-[11px] text-slate-500">nm — restricts X below to the wavelengths in this window</span>
                </div>
              </div>

              {/* column selection */}
              <div className="grid sm:grid-cols-2 gap-4 mb-5">
                <div>
                  <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Independent variables (X) — {nSelectedX} selected</div>
                  <div className="max-h-40 overflow-auto rounded-lg border border-slate-800 p-2 space-y-1">
                    {activePanel.headers.map((h) => (
                      <label key={h} className={`flex items-center gap-2 text-xs px-2 py-1 rounded cursor-pointer ${h === activePanel.yCol ? "opacity-30 pointer-events-none" : "hover:bg-slate-800/60"}`}>
                        <input type="checkbox" checked={activePanel.xCols.includes(h)} onChange={() => toggleX(activeMethod, h)} />
                        <span className="font-mono text-slate-300">{h}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">Dependent variable (Y)</div>
                  <select value={activePanel.yCol || ""} onChange={(e) => updatePanel(activeMethod, { yCol: e.target.value })}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm font-mono text-slate-200">
                    {activePanel.headers.map((h) => <option key={h} value={h}>{h}</option>)}
                  </select>

                  {/* method params */}
                  <div className="mt-4 space-y-3">
                    {(activeMethod === "pls" || activeMethod === "pcr") && (
                      <ParamNumber label="Number of components (Latent Variables)" value={activePanel.params.ncomp}
                        onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, ncomp: v } })} min={1} max={20} />
                    )}
                    {activeMethod === "ann" && (
                      <>
                        <ParamNumber label="Hidden layer neurons" value={activePanel.params.hidden}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, hidden: v } })} min={1} max={30} />
                        <ParamNumber label="Epochs" value={activePanel.params.epochs}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, epochs: v } })} min={50} max={2000} step={50} />
                        <ParamNumber label="Learning rate" value={activePanel.params.lr}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, lr: v } })} min={0.001} max={0.5} step={0.001} />
                      </>
                    )}
                    {activeMethod === "ga" && (
                      <>
                        <ParamNumber label="Population size" value={activePanel.params.popSize}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, popSize: v } })} min={10} max={100} />
                        <ParamNumber label="Generations" value={activePanel.params.generations}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, generations: v } })} min={5} max={200} />
                        <ParamNumber label="Max selected variables" value={activePanel.params.maxVars}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, maxVars: v } })} min={2} max={30} />
                      </>
                    )}
                    {activeMethod === "range" && (
                      <>
                        <ParamNumber label="Number of intervals to test" value={activePanel.params.nIntervals}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, nIntervals: v } })} min={2} max={30} />
                        <ParamNumber label="Max latent variables per interval" value={activePanel.params.maxComp}
                          onChange={(v) => updatePanel(activeMethod, { params: { ...activePanel.params, maxComp: v } })} min={1} max={15} />
                        <p className="text-[11px] text-slate-500 leading-relaxed">
                          Splits the selected X columns into equal-width wavelength intervals, fits PLS with
                          leave-one-out CV in each, and flags the interval with the lowest RMSECV — i.e. the
                          region with the best information content and least noise.
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>

              <button onClick={() => handleRun(activeMethod)} disabled={activePanel.status === "running"}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium text-sm text-white ${m_bar(activeMeta)} disabled:opacity-50`}>
                {activePanel.status === "running" ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                {activePanel.status === "running"
                  ? (activeMethod === "range" ? "Evaluating ranges..." : "Building model...")
                  : (activeMethod === "range" ? "Evaluate ranges" : "Run model")}
              </button>
            </>
          )}

          {/* results */}
          {activePanel.result && activeMethod !== "range" && (
            <div className="mt-6 pt-6 border-t border-slate-800">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
                <Metric label="R² (calibration)" value={fmt(activePanel.result.r2Train)} />
                <Metric label="R² (test)" value={fmt(activePanel.result.r2Test)} highlight />
                <Metric label="RMSE (calibration)" value={fmt(activePanel.result.rmseTrain)} />
                <Metric label="RMSE (test)" value={fmt(activePanel.result.rmseTest)} />
              </div>
              <div className="text-xs text-slate-500 mb-3 font-mono">
                n(train)={activePanel.result.nTrain} · n(test)={activePanel.result.nTest}
                {activePanel.result.model?.kUsed ? ` · latent variables=${activePanel.result.model.kUsed}` : ""}
                {activePanel.result.model?.idxs ? ` · variables selected=${activePanel.result.model.idxs.length}` : ""}
              </div>
              <ChartHeading>Regression plot — experimental vs. calculated concentration</ChartHeading>
              <ResultChart data={activePanel.result.chartData} color={activeMeta} />

              {activePanel.result.rmsecvCurve && activePanel.result.rmsecvCurve.length > 1 && (
                <>
                  <ChartHeading>RMSECV vs. number of latent variables (leave-one-out CV)</ChartHeading>
                  <RmsecvChart curve={activePanel.result.rmsecvCurve} color={activeMeta} />
                </>
              )}
            </div>
          )}

          {/* range-selection results */}
          {activePanel.result && activeMethod === "range" && (
            <div className="mt-6 pt-6 border-t border-slate-800">
              <RangeSelectionResults result={activePanel.result} meta={activeMeta} />
            </div>
          )}
        </div>

        {/* other known methods */}
        <div className="mb-4">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-3">Other known chemometric methods (coming soon)</div>
          <div className="flex flex-wrap gap-2">
            {OTHER_METHODS.map((name) => (
              <span key={name} className="px-3 py-1.5 rounded-full border border-slate-800 bg-slate-900/50 text-xs text-slate-400 font-mono">
                {name}
              </span>
            ))}
          </div>
        </div>

        <p className="text-[11px] text-slate-600 mt-8">
          Note: R² here is approximate (a random 80/20 split each run) for quick exploratory model comparison —
          it is not a substitute for full method validation per ICH Q2.
        </p>
      </div>
    </div>
  );
}

function m_bar(meta) { return meta.bar + " hover:opacity-90"; }

function ParamNumber({ label, value, onChange, min, max, step = 1 }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
        <span>{label}</span>
        <span className="font-mono text-slate-300">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-teal-500" />
    </div>
  );
}

function Metric({ label, value, highlight }) {
  return (
    <div className={`rounded-lg px-3 py-2.5 border ${highlight ? "border-amber-500/30 bg-amber-500/10" : "border-slate-800 bg-slate-900/50"}`}>
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`font-mono text-lg font-semibold ${highlight ? "text-amber-300" : "text-slate-200"}`}>{value}</div>
    </div>
  );
}

function ChartHeading({ children }) {
  return <div className="text-xs uppercase tracking-wide text-slate-500 mb-2 mt-5 first:mt-0">{children}</div>;
}

function RmsecvChart({ curve, color }) {
  const best = curve.reduce((b, p) => (p.rmsecv < b.rmsecv ? p : b), curve[0]);
  return (
    <div>
      <div className="h-64 bg-slate-950/60 rounded-xl border border-slate-800 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curve} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis type="number" dataKey="ncomp" name="Latent variables" domain={["dataMin", "dataMax"]}
              allowDecimals={false} tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "Number of latent variables", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
            <YAxis type="number" dataKey="rmsecv" name="RMSECV" tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "RMSECV (µg/mL)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
              formatter={(v) => fmt(v, 4)} />
            <ReferenceLine x={best.ncomp} stroke="#f59e0b" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="rmsecv" stroke="currentColor" className={color.accent}
              strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="text-xs text-slate-500 mt-2 font-mono">
        Minimum RMSECV = {fmt(best.rmsecv)} at {best.ncomp} latent variable{best.ncomp > 1 ? "s" : ""} (dashed line)
      </div>
    </div>
  );
}


function RangeSelectionResults({ result, meta }) {
  const { baseline, intervals, best } = result;
  return (
    <div>
      <div className="mb-4 px-4 py-3 rounded-lg bg-orange-500/10 border border-orange-500/30 text-sm text-orange-200">
        Recommended range: <b className="font-display">{best.label} nm</b> ({best.nVars} variables,{" "}
        {best.ncomp} latent variable{best.ncomp > 1 ? "s" : ""}) — RMSECV = <span className="font-mono">{fmt(best.rmsecv)}</span>,{" "}
        R² = <span className="font-mono">{fmt(best.r2)}</span>. Full-spectrum baseline: RMSECV ={" "}
        <span className="font-mono">{fmt(baseline.rmsecv)}</span> ({baseline.ncomp} LVs). Noisy / low-information
        regions (higher RMSECV) can be excluded to improve selectivity and sensitivity.
      </div>

      <ChartHeading>RMSECV by wavelength interval (leave-one-out CV)</ChartHeading>
      <div className="h-72 bg-slate-950/60 rounded-xl border border-slate-800 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={intervals} margin={{ top: 10, right: 20, bottom: 45, left: 0 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} angle={-40} textAnchor="end" interval={0} height={60}
              label={{ value: "Wavelength range (nm)", position: "insideBottom", offset: -40, fill: "#64748b", fontSize: 11 }} />
            <YAxis tick={{ fill: "#64748b", fontSize: 11 }}
              label={{ value: "RMSECV (µg/mL)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} formatter={(v) => fmt(v, 4)} />
            <ReferenceLine y={baseline.rmsecv} stroke="#64748b" strokeDasharray="4 4"
              label={{ value: "full spectrum", position: "right", fill: "#94a3b8", fontSize: 10 }} />
            <Bar dataKey="rmsecv" radius={[4, 4, 0, 0]}>
              {intervals.map((entry, i) => (
                <Cell key={i} fill={entry.label === best.label ? "#f59e0b" : "#fb923c"} fillOpacity={entry.label === best.label ? 1 : 0.55} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <ChartHeading>Ranked intervals</ChartHeading>
      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full text-xs font-mono">
          <thead className="bg-slate-800/60 text-slate-400">
            <tr>
              <th className="px-3 py-2 text-left">Range (nm)</th>
              <th className="px-3 py-2 text-left">Variables</th>
              <th className="px-3 py-2 text-left">Latent variables</th>
              <th className="px-3 py-2 text-left">RMSECV</th>
              <th className="px-3 py-2 text-left">R²</th>
            </tr>
          </thead>
          <tbody>
            {result.ranked.map((it) => (
              <tr key={it.label} className={`border-t border-slate-800/60 ${it.label === best.label ? "bg-amber-500/10 text-amber-200" : "text-slate-400"}`}>
                <td className="px-3 py-1.5">{it.label}</td>
                <td className="px-3 py-1.5">{it.nVars}</td>
                <td className="px-3 py-1.5">{it.ncomp}</td>
                <td className="px-3 py-1.5">{fmt(it.rmsecv)}</td>
                <td className="px-3 py-1.5">{fmt(it.r2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


function ResultChart({ data, color }) {
  const all = data.flatMap((d) => [d.actual, d.predicted]).filter(Number.isFinite);
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 || 1;
  const domain = [lo - pad, hi + pad];
  return (
    <div className="h-72 bg-slate-950/60 rounded-xl border border-slate-800 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
          <CartesianGrid stroke="#1e293b" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={domain} tick={{ fill: "#64748b", fontSize: 11 }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={domain} tick={{ fill: "#64748b", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
          <ReferenceLine segment={[{ x: domain[0], y: domain[0] }, { x: domain[1], y: domain[1] }]} stroke="#475569" strokeDasharray="4 4" />
          <Scatter data={data} fill="currentColor" className={color.accent} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
