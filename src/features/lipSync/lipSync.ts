// LipSync.ts
import { LipSyncAnalyzeResult } from "./lipSyncAnalyzeResult";

export type VisemeWeights = { a:number; e:number; u:number; o:number; rest:number };

const LEN = 2048;
const ATTACK = 0.06;   // s
const RELEASE = 0.12;  // s

export class LipSync {
  public readonly audio: AudioContext;
  private analyser: AnalyserNode;
  private bp: BiquadFilterNode;
  private hp: BiquadFilterNode;
  private comp: DynamicsCompressorNode;
  private gainTap: GainNode;
  private tPrev = 0;
  private env = 0; // smoothed envelope [0..1]
  private timeBuf = new Float32Array(LEN);
  private freqBuf = new Float32Array(LEN);
  private _latency = 0.16; // seconds, tweak per TTS/ASR path

  constructor(audio: AudioContext, latencySeconds = 0.16) {
    this.audio = audio;
    this._latency = latencySeconds;

    // analysis chain: source -> hp -> bp -> comp -> analyser (+ tap to destination outside)
    this.hp = audio.createBiquadFilter(); this.hp.type = "highpass"; this.hp.frequency.value = 90;
    this.bp = audio.createBiquadFilter(); this.bp.type = "bandpass"; this.bp.frequency.value = 1200; this.bp.Q.value = 0.7;
    this.comp = audio.createDynamicsCompressor(); this.comp.threshold.value = -30; this.comp.knee.value = 20; this.comp.ratio.value = 3;
    this.analyser = audio.createAnalyser(); this.analyser.fftSize = LEN; this.analyser.smoothingTimeConstant = 0.5;

    this.gainTap = audio.createGain(); // for app to route to speakers if desired
    this.tPrev = audio.currentTime;
  }

  /** Connect an AudioNode as source (BufferSource, MediaStreamSource, etc.) */
  connectSource(node: AudioNode) {
    node.connect(this.hp);
    this.hp.connect(this.bp);
    this.bp.connect(this.comp);
    this.comp.connect(this.analyser);
    // let caller decide whether/where to hear the audio:
    this.comp.connect(this.gainTap);
  }

  /** route to output if you want to hear it */
  connectToDestination() { this.gainTap.connect(this.audio.destination); }

  setLatencySeconds(v: number){ this._latency = Math.max(0, v); }

  /** Core update; call every frame. Returns envelope + coarse visemes + clock time. */
  update(): LipSyncAnalyzeResult & { visemes: VisemeWeights; tAudio: number } {
    const tNow = this.audio.currentTime;
    const dt = Math.max(1/120, Math.min(1/20, tNow - this.tPrev));
    this.tPrev = tNow;

    this.analyser.getFloatTimeDomainData(this.timeBuf);

    // RMS envelope
    let sum = 0;
    for (let i = 0; i < LEN; i++) sum += this.timeBuf[i] * this.timeBuf[i];
    const rms = Math.sqrt(sum / LEN); // ~0..1
    const target = 1 / (1 + Math.exp(-35 * rms + 5)); // soft companding  (similar shape, gentler)

    // attack/release smoothing
    const coef = target > this.env
      ? 1 - Math.exp(-dt / ATTACK)
      : 1 - Math.exp(-dt / RELEASE);
    this.env += (target - this.env) * coef;

    // basic vowel classification via spectral centroid
    this.analyser.getFloatFrequencyData(this.freqBuf);
    let num = 0, den = 0;
    const nyquist = this.audio.sampleRate / 2;
    for (let i = 0; i < this.analyser.frequencyBinCount; i++) {
      const mag = Math.pow(10, this.freqBuf[i] / 20); // linear magnitude
      const f = (i / this.analyser.frequencyBinCount) * nyquist;
      num += f * mag; den += mag;
    }
    const centroid = (den > 0 ? num / den : 0); // Hz

    // crude vowel mapping: (A~800Hz, E~2300, U~350, O~500). We blend softly.
    const w = (f:number, c:number, s:number)=>Math.exp(-0.5*Math.pow((f-c)/s,2));
    const a = w(centroid, 800, 500);
    const e = w(centroid, 2300, 900);
    const u = w(centroid, 350, 250);
    const o = w(centroid, 500, 300);

    // normalize visemes and scale by envelope
    let sumV = a+e+u+o + 1e-6;
    const vis: VisemeWeights = {
      a: this.env * (a/sumV),
      e: this.env * (e/sumV),
      u: this.env * (u/sumV),
      o: this.env * (o/sumV),
      rest: Math.max(0, 1 - this.env)
    };

    return { volume: this.env, visemes: vis, tAudio: tNow + this._latency };
  }

  /** Helpers similar to existing API */
  async playFromArrayBuffer(buf: ArrayBuffer, onEnded?: ()=>void) {
    const audioBuf = await this.audio.decodeAudioData(buf);
    const src = this.audio.createBufferSource();
    src.buffer = audioBuf;
    this.connectSource(src);
    this.connectToDestination(); // optional: remove if you don’t want to hear it
    src.start();
    if (onEnded) src.addEventListener("ended", onEnded);
  }

  async playFromURL(url: string, onEnded?: ()=>void) {
    const res = await fetch(url); const buf = await res.arrayBuffer();
    return this.playFromArrayBuffer(buf, onEnded);
  }
}