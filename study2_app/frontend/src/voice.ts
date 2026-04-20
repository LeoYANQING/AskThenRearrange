// STT helpers — browser-side recording + upload to /voice/stt (Dashscope paraformer).

export class Recorder {
  private ctx?: AudioContext
  private stream?: MediaStream
  private source?: MediaStreamAudioSourceNode
  private processor?: ScriptProcessorNode
  private gain?: GainNode
  private chunks: Float32Array[] = []
  private sampleRate = 0
  private startedAt = 0
  recording = false

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    const AC = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext
    this.ctx = new AC()
    this.sampleRate = this.ctx.sampleRate
    this.source = this.ctx.createMediaStreamSource(this.stream)
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1)
    this.gain = this.ctx.createGain()
    this.gain.gain.value = 0
    this.chunks = []
    this.processor.onaudioprocess = (e) => {
      this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    this.source.connect(this.processor)
    this.processor.connect(this.gain)
    this.gain.connect(this.ctx.destination)
    this.startedAt = performance.now()
    this.recording = true
  }

  async stop(): Promise<{ blob: Blob; durationMs: number; sampleRate: number; peak: number }> {
    if (!this.recording) throw new Error('not recording')
    this.processor?.disconnect()
    this.source?.disconnect()
    this.gain?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    await this.ctx?.close()
    this.recording = false

    const durationMs = performance.now() - this.startedAt
    const pcm = flatten(this.chunks)
    let peak = 0
    for (let i = 0; i < pcm.length; i++) {
      const a = Math.abs(pcm[i])
      if (a > peak) peak = a
    }
    const wav = encodeWAV(pcm, this.sampleRate)
    const blob = new Blob([wav], { type: 'audio/wav' })
    return { blob, durationMs, sampleRate: this.sampleRate, peak }
  }
}

export async function transcribe(
  blob: Blob,
  language: 'zh' | 'en' | 'ja' | 'ko' | 'auto' = 'zh',
  sampleRate = 16000,
): Promise<string> {
  const fd = new FormData()
  fd.append('file', blob, 'recording.wav')
  fd.append('language', language)
  fd.append('sample_rate', String(sampleRate))
  const r = await fetch('/voice/stt', { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`STT HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`)
  const j = await r.json()
  return j.text ?? ''
}

function flatten(chunks: Float32Array[]): Float32Array {
  let total = 0
  chunks.forEach((c) => (total += c.length))
  const out = new Float32Array(total)
  let off = 0
  chunks.forEach((c) => {
    out.set(c, off)
    off += c.length
  })
  return out
}

function encodeWAV(pcm: Float32Array, sr: number): ArrayBuffer {
  const buf = new ArrayBuffer(44 + pcm.length * 2)
  const v = new DataView(buf)
  const writeStr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  v.setUint32(4, 36 + pcm.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  v.setUint32(16, 16, true)
  v.setUint16(20, 1, true)
  v.setUint16(22, 1, true)
  v.setUint32(24, sr, true)
  v.setUint32(28, sr * 2, true)
  v.setUint16(32, 2, true)
  v.setUint16(34, 16, true)
  writeStr(36, 'data')
  v.setUint32(40, pcm.length * 2, true)
  let o = 44
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    o += 2
  }
  return buf
}
