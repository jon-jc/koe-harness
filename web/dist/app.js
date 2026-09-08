"use strict";(()=>{var h=`
class PCMDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetSampleRate || 16000;
    this.frameSamples = Math.round(this.targetRate * (opts.frameMs || 20) / 1000);
    this.ratio = sampleRate / this.targetRate;
    this.buffer = new Int16Array(this.frameSamples);
    this.filled = 0;
    this.cursor = 0;
    this.peak = 0;
    this.sinceLevel = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    // Decimate by averaging over each source window. Averaging is a crude
    // low-pass, but dropping samples outright aliases high-frequency content
    // down into the speech band, which sounds like a metallic buzz and
    // measurably degrades recognition.
    while (this.cursor < channel.length) {
      const start = this.cursor;
      const end = Math.min(channel.length, start + this.ratio);
      let sum = 0;
      let count = 0;
      for (let i = Math.floor(start); i < Math.ceil(end) && i < channel.length; i++) {
        sum += channel[i];
        count++;
      }
      const value = count > 0 ? sum / count : 0;
      const magnitude = Math.abs(value);
      if (magnitude > this.peak) this.peak = magnitude;

      const clamped = Math.max(-1, Math.min(1, value));
      this.buffer[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this.filled >= this.frameSamples) {
        // Copy: the buffer is reused immediately, and a transferred view of it
        // would be detached out from under the next frame.
        const frame = this.buffer.slice(0);
        this.port.postMessage({ type: 'frame', frame: frame.buffer }, [frame.buffer]);
        this.filled = 0;
      }
      this.cursor = end;
    }
    this.cursor -= channel.length;

    this.sinceLevel += channel.length;
    if (this.sinceLevel >= sampleRate / 20) {
      this.port.postMessage({ type: 'level', level: this.peak });
      this.peak = 0;
      this.sinceLevel = 0;
    }
    return true;
  }
}
registerProcessor('pcm-downsampler', PCMDownsampler);
`,a=class{constructor(t){this.options=t;this.context=null;this.node=null;this.source=null;this.stream=null;this.moduleUrl=null}get active(){return this.context!==null}async start(){if(this.context)return;this.stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:!0,noiseSuppression:!0,autoGainControl:!0}}),this.context=new AudioContext;let t=new Blob([h],{type:"application/javascript"});this.moduleUrl=URL.createObjectURL(t),await this.context.audioWorklet.addModule(this.moduleUrl),this.node=new AudioWorkletNode(this.context,"pcm-downsampler",{numberOfInputs:1,numberOfOutputs:0,processorOptions:{targetSampleRate:this.options.targetSampleRate??16e3,frameMs:this.options.frameMs??20}}),this.node.port.onmessage=e=>{let s=e.data;s.type==="frame"&&s.frame?this.options.onFrame(s.frame):s.type==="level"&&this.options.onLevel&&this.options.onLevel(s.level??0)},this.source=this.context.createMediaStreamSource(this.stream),this.source.connect(this.node)}async stop(){this.node?.port.close(),this.source?.disconnect(),this.node?.disconnect(),this.stream?.getTracks().forEach(t=>t.stop()),this.context&&await this.context.close(),this.moduleUrl&&URL.revokeObjectURL(this.moduleUrl),this.node=null,this.source=null,this.stream=null,this.context=null,this.moduleUrl=null}};var o=class{constructor(t,e){this.url=t;this.handlers=e;this.socket=null;this.dropped=0;this.sent=0}get connected(){return this.socket?.readyState===WebSocket.OPEN}get droppedFrames(){return this.dropped}get sentFrames(){return this.sent}connect(){return new Promise((t,e)=>{let s=new WebSocket(this.url);s.binaryType="arraybuffer",this.socket=s,s.onopen=()=>{this.handlers.onOpen?.(),t()},s.onmessage=i=>{if(typeof i.data=="string")try{this.handlers.onMessage(JSON.parse(i.data))}catch{this.handlers.onError?.("received a malformed frame from the server")}},s.onerror=()=>{this.handlers.onError?.("connection failed"),e(new Error("websocket error"))},s.onclose=i=>{this.socket=null,this.handlers.onClose?.(i)}})}start(t){this.send({type:"start",language:t.language,partial_interval_ms:t.partialIntervalMs??500})}sendAudio(t){if(!(!this.connected||!this.socket)){if(this.socket.bufferedAmount>524288){this.dropped+=1;return}this.socket.send(t),this.sent+=1}}stop(){this.send({type:"stop"})}close(){this.socket?.close(),this.socket=null}send(t){!this.connected||!this.socket||this.socket.send(JSON.stringify(t))}};var p={ja:{idle:"\u5F85\u6A5F\u4E2D",connecting:"\u63A5\u7D9A\u4E2D\u2026",listening:"\u8A8D\u8B58\u4E2D",stopped:"\u505C\u6B62\u3057\u307E\u3057\u305F",start:"\u9332\u97F3\u958B\u59CB",stop:"\u505C\u6B62",empty:"\u30DE\u30A4\u30AF\u3092\u958B\u59CB\u3059\u308B\u3068\u3001\u3053\u3053\u306B\u6587\u5B57\u8D77\u3053\u3057\u304C\u8868\u793A\u3055\u308C\u307E\u3059\u3002",speaking:"\u767A\u8A71\u4E2D",silence:"\u7121\u97F3"},en:{idle:"idle",connecting:"connecting\u2026",listening:"listening",stopped:"stopped",start:"Start",stop:"Stop",empty:"Start the microphone and the transcript will appear here.",speaking:"speech",silence:"silence"}};function n(r){let t=document.getElementById(r);if(!t)throw new Error(`missing element #${r}`);return t}var c=class{constructor(){this.client=null;this.capture=null;this.finals=[];this.committed="";this.pending="";this.language="ja";this.running=!1;this.startedAt=0;this.toggle=n("toggle");this.status=n("status");this.provider=n("provider");this.meter=n("meter-fill");this.transcript=n("transcript");this.stats=n("stats");this.languageSelect=n("language");this.speech=n("speech-state");this.toggle.addEventListener("click",()=>void this.onToggle()),this.languageSelect.addEventListener("change",()=>{this.language=this.languageSelect.value,this.render()}),this.render()}get strings(){return this.language==="en"?p.en:p.ja}async onToggle(){this.running?await this.stop():await this.start()}async start(){this.finals=[],this.committed="",this.pending="",this.setStatus(this.strings.connecting);let t=location.protocol==="https:"?"wss":"ws";this.client=new o(`${t}://${location.host}/v1/stream`,{onMessage:e=>this.onMessage(e),onError:e=>this.setStatus(`error: ${e}`),onClose:()=>{this.running&&this.stop()}});try{await this.client.connect()}catch{this.setStatus("could not connect");return}this.client.start({language:this.language,partialIntervalMs:400}),this.capture=new a({onFrame:e=>this.client?.sendAudio(e),onLevel:e=>{this.meter.style.width=`${Math.min(100,e*140)}%`}});try{await this.capture.start()}catch{this.setStatus("microphone unavailable \u2014 check permissions"),this.client.close();return}this.running=!0,this.startedAt=performance.now(),this.toggle.textContent=this.strings.stop,this.toggle.classList.add("recording"),this.setStatus(this.strings.listening),this.render()}async stop(){this.running=!1,this.toggle.textContent=this.strings.start,this.toggle.classList.remove("recording"),this.setStatus(this.strings.stopped),this.meter.style.width="0%",await this.capture?.stop(),this.capture=null,this.client?.stop(),setTimeout(()=>this.client?.close(),1500)}onMessage(t){switch(t.type){case"started":this.provider.textContent=t.provider;break;case"speech":this.speech.textContent=t.state==="start"?this.strings.speaking:this.strings.silence,this.speech.className=t.state==="start"?"badge active":"badge";break;case"partial":this.committed=t.committed,this.pending=t.pending,this.render();break;case"final":this.finals.push({text:t.text,start:t.start,end:t.end,speaker:t.speaker}),this.committed="",this.pending="",this.render();break;case"transcript":this.stats.textContent=`${t.duration.toFixed(1)}s \xB7 $${t.cost_usd.toFixed(5)} \xB7 ${this.client?.sentFrames??0} frames sent, ${this.client?.droppedFrames??0} dropped`;break;case"error":this.setStatus(`error: ${t.message}`);break}}setStatus(t){this.status.textContent=t}render(){if(this.toggle.textContent=this.running?this.strings.stop:this.strings.start,this.finals.length===0&&!this.committed&&!this.pending){this.transcript.innerHTML=`<p class="empty">${this.strings.empty}</p>`;return}let t=[];for(let e of this.finals){let s=e.speaker?`<span class="speaker">${l(e.speaker)}</span>`:"";t.push(`<p class="line final">${s}<span class="text">${l(e.text)}</span><span class="time">${e.start.toFixed(1)}s</span></p>`)}if((this.committed||this.pending)&&t.push(`<p class="line interim"><span class="text">${l(this.committed)}<span class="pending">${l(this.pending)}</span></span></p>`),this.transcript.innerHTML=t.join(""),this.transcript.scrollTop=this.transcript.scrollHeight,this.running){let e=(performance.now()-this.startedAt)/1e3;this.stats.textContent=`${e.toFixed(1)}s \xB7 ${this.finals.length} segments`}}};function l(r){return r.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;")}document.addEventListener("DOMContentLoaded",()=>{new c});})();
//# sourceMappingURL=app.js.map
