import os
import uuid
from typing import Dict

from flask import Flask, request, jsonify, send_from_directory, abort, Response

try:
    from werkzeug.utils import secure_filename
except Exception:
    def secure_filename(name: str) -> str:
        name = (name or "").strip().replace("\x00", "")
        name = name.replace("/", "_").replace("\\", "_")
        return "".join(c for c in name if c.isalnum() or c in {".", "_", "-"})

from pipeline import generate_song, VALID_STYLES, midi_to_mp3, midi_to_musicxml, generate_melody_midi


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mid", ".midi"}

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    template_folder=BASE_DIR,
)

_ASSETS: Dict[str, Dict[str, str]] = {}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> str:
    if file_storage is None:
        raise ValueError("Missing file field.")
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Empty filename.")
    if not allowed_file(filename):
        raise ValueError("Invalid file type. Please upload a .mid or .midi file.")
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    file_storage.save(dst)
    return dst


def _to_download_url(abs_path: str) -> str:
    return "/download/" + os.path.basename(abs_path)


def _safe_in_output(p: str) -> bool:
    if not p:
        return False
    p_abs = os.path.abspath(p)
    out_abs = os.path.abspath(OUTPUT_DIR)
    return p_abs.startswith(out_abs) and os.path.exists(p_abs)


def _pick_asset(asset_rec: Dict[str, str]) -> str:
    want = (request.args.get("type") or "").strip().lower()
    accept = (request.headers.get("Accept") or "").lower()

    midi_path = asset_rec.get("midi_path", "")
    mp3_path = asset_rec.get("mp3_path", "")
    xml_path = asset_rec.get("musicxml_path", "")

    if want == "audio":
        if _safe_in_output(mp3_path):
            return mp3_path
        if _safe_in_output(midi_path):
            new_mp3 = midi_to_mp3(midi_path)
            if new_mp3 and _safe_in_output(new_mp3):
                asset_rec["mp3_path"] = new_mp3
                return new_mp3
        if _safe_in_output(midi_path):
            return midi_path

    if want == "score":
        if _safe_in_output(xml_path):
            return xml_path
        if _safe_in_output(midi_path):
            new_xml = midi_to_musicxml(midi_path)
            if new_xml and _safe_in_output(new_xml):
                asset_rec["musicxml_path"] = new_xml
                return new_xml
        if _safe_in_output(midi_path):
            return midi_path

    if "audio" in accept:
        if _safe_in_output(mp3_path):
            return mp3_path
        if _safe_in_output(midi_path):
            new_mp3 = midi_to_mp3(midi_path)
            if new_mp3 and _safe_in_output(new_mp3):
                asset_rec["mp3_path"] = new_mp3
                return new_mp3
        if _safe_in_output(midi_path):
            return midi_path

    if "xml" in accept or "musicxml" in accept:
        if _safe_in_output(xml_path):
            return xml_path
        if _safe_in_output(midi_path):
            new_xml = midi_to_musicxml(midi_path)
            if new_xml and _safe_in_output(new_xml):
                asset_rec["musicxml_path"] = new_xml
                return new_xml
        if _safe_in_output(midi_path):
            return midi_path

    if _safe_in_output(midi_path):
        return midi_path
    if _safe_in_output(mp3_path):
        return mp3_path
    if _safe_in_output(xml_path):
        return xml_path

    abort(404)


def _serve_html_with_patch(filename: str, inject_patch: bool = False) -> Response:
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        abort(404)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    tag = '<script src="/_ui_patch.js"></script>'
    if tag not in html:
        if "</body>" in html:
            html = html.replace("</body>", tag + "</body>")
        else:
            html = html + tag

    resp = Response(html, mimetype="text/html; charset=utf-8")
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/_patch_alive")
def patch_alive():
    return "", 204

@app.route("/_ui_patch.js")
def ui_patch_js():
    js = r"""
(() => {
  const KEY = '__ui_patch_melody_final_v1__';
  if (window[KEY]) return;
  window[KEY] = true;

  const PENDING_KEY = '__melody_pending__';
  const STARTED_KEY = '__melody_thinking_started__';

  function isThinkingMelodyPage(){
    try{ return (location.pathname || '').toLowerCase().includes('thinking_melody'); }catch(e){}
    return false;
  }

  function isVisible(el){
    if(!el) return false;
    try{
      const st = getComputedStyle(el);
      if(st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
      const r = el.getBoundingClientRect();
      return r.width > 10 && r.height > 10;
    }catch(e){}
    return false;
  }

  function readVal(sel, fallback){
    try{
      const el = document.querySelector(sel);
      const v = el ? (el.value || '').trim() : '';
      return v || fallback;
    }catch(e){}
    return fallback;
  }

  function hasKeyMeterControls(){
    try{
      const k = document.querySelector('[name="key"],#key,[name="noteName"],#noteName');
      const m = document.querySelector('[name="meter"],#meter,[name="timeSig"],#timeSig');
      return !!(k && m);
    }catch(e){}
    return false;
  }

  function melodySelected(){
    try{
      if (isThinkingMelodyPage()) return true;

      const qs = new URLSearchParams(location.search || '');
      const qm = (qs.get('mode') || '').toLowerCase();
      if (qm.includes('melody')) return true;

      try{
        const sm = (sessionStorage.getItem('mode') || '').toLowerCase();
        if (sm.includes('melody')) return true;
      }catch(e){}
      try{
        const lm = (localStorage.getItem('mode') || '').toLowerCase();
        if (lm.includes('melody')) return true;
      }catch(e){}

      if (hasKeyMeterControls()) return true;

      const r = document.querySelector('input[name="mode"]:checked');
      if (r && ((r.value || '') + '').toLowerCase().includes('melody')) return true;

      const s = document.querySelector('select[name="mode"], #mode, #modeSelect, #mode-select');
      if (s && ((s.value || '') + '').toLowerCase().includes('melody')) return true;

      const active = document.querySelector('[data-mode="melody"].active, [data-mode="melody"][aria-pressed="true"], [data-mode="melody"][aria-selected="true"]');
      if (active) return true;
    }catch(e){}
    return false;
  }

  function looksGenerate(el){
    try{
      const id  = (el.id || '').toLowerCase();
      const cls = (el.className || '').toString().toLowerCase();
      const da  = ((el.getAttribute && el.getAttribute('data-action')) || '').toLowerCase();
      const dt  = ((el.getAttribute && el.getAttribute('data-testid')) || '').toLowerCase();
      const txt = ((el.innerText || el.textContent || '') + '').trim().toLowerCase();

      return da === 'generate'
        || dt.includes('generate')
        || id === 'gen'
        || id.includes('generate')
        || cls.includes('generate')
        || txt.includes('generate')
        || txt.includes('生成');
    }catch(e){}
    return false;
  }

  function findGenerate(){
    const ids = ['gen','generate','generateBtn','generateButton','btnGenerate','btn_generate'];
    for(const id of ids){
      const el = document.getElementById(id);
      if(el && isVisible(el)) return el;
    }
    try{
      const list = document.querySelectorAll('button,[role="button"],input[type="button"],input[type="submit"],a,div,span');
      for(const el of list){
        if(looksGenerate(el) && isVisible(el)) return el;
      }
    }catch(e){}
    return null;
  }

  function enable(el){
    try{
      if(!el) return;
      try{ el.disabled = false; }catch(e){}
      try{ el.removeAttribute('disabled'); }catch(e){}
      try{ if(el.getAttribute('aria-disabled') === 'true') el.setAttribute('aria-disabled','false'); }catch(e){}
      try{ el.style.pointerEvents = 'auto'; }catch(e){}
      try{ if(el.parentElement) el.parentElement.style.pointerEvents = 'auto'; }catch(e){}
    }catch(e){}
  }

  function setPendingFromUI(){
    const req = {
      mode: 'melody',
      key: readVal('[name="key"],#key,[name="noteName"],#noteName', 'C'),
      meter: readVal('[name="meter"],#meter,[name="timeSig"],#timeSig', '4/4'),
      seed: readVal('[name="seed"],#seed,[name="custom_seed"],#custom_seed', ''),
      ts: Date.now()
    };
    try{
      sessionStorage.setItem('mode','melody');
      sessionStorage.setItem(PENDING_KEY, JSON.stringify(req));
      sessionStorage.removeItem(STARTED_KEY);
    }catch(e){}
  }

  async function runThinkingIfPending(){
    if (!isThinkingMelodyPage()) return;

    try{
      if (sessionStorage.getItem(STARTED_KEY) === '1') return;
    }catch(e){}

    let req = null;
    try{ req = JSON.parse(sessionStorage.getItem(PENDING_KEY) || 'null'); }catch(e){ req = null; }

    if (!req || req.mode !== 'melody'){
      setTimeout(() => {
        try{
          let r2 = null;
          try{ r2 = JSON.parse(sessionStorage.getItem(PENDING_KEY) || 'null'); }catch(e){ r2 = null; }
          if(!r2){ location.href = '/input.html'; }
        }catch(e){}
      }, 3000);
      return;
    }

    try{
      sessionStorage.setItem(STARTED_KEY, '1');
      sessionStorage.removeItem(PENDING_KEY);
    }catch(e){}

    await new Promise(r => setTimeout(r, 80));

    const fd = new FormData();
    fd.append('mode','melody');
    fd.append('key', req.key || 'C');
    fd.append('meter', req.meter || '4/4');
    if (req.seed) fd.append('seed', req.seed);

    const res = await fetch('/api/generate', { method:'POST', body: fd });
    const data = await res.json().catch(() => ({}));
    if (!data || data.success !== true){
      try{ location.href = '/input.html'; }catch(e){}
      return;
    }

    try{
      sessionStorage.setItem('mode','melody');
      if (data.file_url) sessionStorage.setItem('file_url', data.file_url);
      if (data.midi_path) sessionStorage.setItem('midi_path', data.midi_path);
      if (data.audio_url) sessionStorage.setItem('audio_url', data.audio_url);
      if (data.score_url) sessionStorage.setItem('score_url', data.score_url);
      if (data.asset_id) sessionStorage.setItem('asset_id', data.asset_id);
    }catch(e){}

    const qs = new URLSearchParams();
    qs.set('mode','melody');
    if (data.file_url) qs.set('file', data.file_url);
    if (data.midi_path) qs.set('midi', data.midi_path);
    if (data.audio_url) qs.set('audio', data.audio_url);
    if (data.score_url) qs.set('score', data.score_url);

    location.href = '/play.html?' + qs.toString();
  }

  function interceptGenerateToThinking(e){
    if (isThinkingMelodyPage()) return;
    if (!melodySelected()) return;

    const gen = findGenerate();
    if (!gen) return;

    let hit = false;
    try{
      const r = gen.getBoundingClientRect();
      hit = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    }catch(err){}

    if (!hit) return;

    e.preventDefault();
    e.stopImmediatePropagation();

    enable(gen);
    setPendingFromUI();
    location.href = '/thinking_melody.html';
  }

  function run(){
    try{
      const gen = findGenerate();
      if (gen) enable(gen);
    }catch(e){}
    runThinkingIfPending();
  }

  document.addEventListener('pointerdown', interceptGenerateToThinking, true);
  document.addEventListener('click', interceptGenerateToThinking, true);

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', run, { once: true });
  }else{
    run();
  }
  setTimeout(run, 80);
})();
"""
    resp = Response(js, mimetype="application/javascript; charset=utf-8")
    resp.headers["Cache-Control"] = "no-store"
    return resp



@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
@app.route("/index.html")
def index_page():
    return _serve_html_with_patch("index.html", inject_patch=True)


@app.route("/input")
@app.route("/input.html")
def input_page():
    return _serve_html_with_patch("input.html", inject_patch=True)


@app.route("/thinking")
@app.route("/thinking.html")
def thinking_page():
    return _serve_html_with_patch("thinking.html", inject_patch=True)


@app.route("/thinking_melody")
@app.route("/thinking_melody.html")
def thinking_melody_page():
    return _serve_html_with_patch("thinking_melody.html", inject_patch=True)


@app.route("/play")
@app.route("/play.html")
def play_page():
    return _serve_html_with_patch("play.html", inject_patch=True)


@app.route("/score")
@app.route("/score.html")
def score_page():
    return _serve_html_with_patch("score.html", inject_patch=True)


@app.route("/api/styles", methods=["GET"])
def api_styles():
    return jsonify({"success": True, "styles": VALID_STYLES})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        form = request.form
        mode = (form.get("mode") or "chord").strip().lower()

        if mode == "melody":
            key = (form.get("key") or form.get("noteName") or "C").strip()
            meter = (form.get("meter") or form.get("timeSig") or "4/4").strip()
            seed_raw = (form.get("seed") or form.get("custom_seed") or "").strip()
            seed = int(seed_raw) if seed_raw.isdigit() else None

            midi_abs = generate_melody_midi(key=key, meter=meter, seed=seed)

            asset_id = uuid.uuid4().hex
            _ASSETS[asset_id] = {"midi_path": midi_abs, "mp3_path": "", "musicxml_path": ""}

            midi_url = _to_download_url(midi_abs)
            base = os.path.splitext(os.path.basename(midi_abs))[0]

            return jsonify({
                "success": True,
                "message": "Generation successful! Please click to play or download.",
                "asset_id": asset_id,
                "midi_path": midi_url,
                "file_url": "/asset/" + asset_id,
                "audio_url": "/download/" + base + ".mp3",
                "score_url": "/download/" + base + ".musicxml",
                "mode": mode,
            })

        file = request.files.get("file")
        if file is None:
            raise ValueError("Missing file field.")
        style = (form.get("style") or "Auto").strip()
        if style not in VALID_STYLES:
            style = "Auto"

        uploaded_file_path = save_upload(file)
        gen = generate_song(uploaded_file_path, selected_style=style, mode=mode)

        if isinstance(gen, dict):
            if not gen.get("success"):
                raise ValueError(gen.get("message") or "Generation failed")
            midi_abs = gen.get("midi_path") or ""
            mp3_abs = gen.get("mp3_path") or ""
            xml_abs = gen.get("musicxml_path") or ""
            chord_preview = gen.get("chord_preview") or []
            detected_style = gen.get("detected_style") or ""
            final_style = gen.get("final_style") or ""
            tempo = gen.get("tempo") or ""
        else:
            midi_abs = str(gen or "")
            mp3_abs = ""
            xml_abs = ""
            chord_preview = []
            detected_style = ""
            final_style = ""
            tempo = ""

        if not midi_abs:
            raise ValueError("Generation failed: missing midi output")

        asset_id = uuid.uuid4().hex
        _ASSETS[asset_id] = {"midi_path": midi_abs, "mp3_path": mp3_abs, "musicxml_path": xml_abs}

        base = os.path.splitext(os.path.basename(midi_abs))[0]
        return jsonify({
            "success": True,
            "message": "Generation successful! Please click to play or download.",
            "asset_id": asset_id,
            "midi_path": _to_download_url(midi_abs),
            "file_url": "/asset/" + asset_id,
            "audio_url": "/download/" + base + ".mp3",
            "score_url": "/download/" + base + ".musicxml",
            "chord_preview": chord_preview,
            "detected_style": detected_style,
            "final_style": final_style,
            "tempo": tempo,
            "mode": mode,
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400


@app.route("/asset/<asset_id>")
def asset(asset_id: str):
    asset_rec = _ASSETS.get(asset_id)
    if not asset_rec:
        abort(404)
    path = _pick_asset(asset_rec)
    if not _safe_in_output(path):
        abort(404)
    filename = os.path.basename(path)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/download/<path:filename>")
def download_file(filename: str):
    filename = secure_filename(filename)
    base, ext = os.path.splitext(filename)
    ext_l = ext.lower()

    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        if ext_l == ".mp3":
            cand1 = os.path.join(OUTPUT_DIR, base + ".mid")
            cand2 = os.path.join(OUTPUT_DIR, base + ".midi")
            midi_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
            if midi_path:
                try:
                    new_mp3 = midi_to_mp3(midi_path)
                    if new_mp3 and os.path.exists(new_mp3):
                        filename = os.path.basename(new_mp3)
                        path = os.path.join(OUTPUT_DIR, filename)
                except Exception:
                    return "", 404
            else:
                return "", 404

        elif ext_l in {".xml", ".musicxml"}:
            cand1 = os.path.join(OUTPUT_DIR, base + ".mid")
            cand2 = os.path.join(OUTPUT_DIR, base + ".midi")
            midi_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
            if midi_path:
                try:
                    new_xml = midi_to_musicxml(midi_path)
                    if new_xml and os.path.exists(new_xml):
                        filename = os.path.basename(new_xml)
                        path = os.path.join(OUTPUT_DIR, filename)
                except Exception:
                    return "", 404
            else:
                return "", 404
        else:
            return "", 404

    if ext_l == ".mp3":
        resp = send_from_directory(OUTPUT_DIR, os.path.basename(path), as_attachment=False, mimetype="audio/mpeg")
        resp.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(path)}"'
        resp.headers["Cache-Control"] = "no-store"
        return resp

    if ext_l in {".xml", ".musicxml"}:
        resp = send_from_directory(
            OUTPUT_DIR,
            os.path.basename(path),
            as_attachment=False,
            mimetype="application/vnd.recordare.musicxml+xml",
        )
        resp.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(path)}"'
        resp.headers["Cache-Control"] = "no-store"
        return resp

    resp = send_from_directory(OUTPUT_DIR, os.path.basename(path), as_attachment=False)
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
