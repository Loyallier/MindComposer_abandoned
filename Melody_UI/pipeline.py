import os
import time
import uuid
import shutil
import logging
import subprocess
import inspect
import random
import sys
import types
import re

from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

VALID_STYLES: List[str] = [
    "Auto",
    "Pop Ballad",
    "Waltz",
    "March",
    "Jazz Swing",
    "EDM",
    "Metal",
    "Sparse_Arpeggio",
]

_STYLE_ALIASES: Dict[str, str] = {
    "auto": "Auto",
    "pop": "Pop Ballad",
    "pop ballad": "Pop Ballad",
    "r&b": "Pop Ballad",
    "rb": "Pop Ballad",
    "lofi": "Pop Ballad",
    "lo-fi": "Pop Ballad",
    "lo fi": "Pop Ballad",
    "classical": "Waltz",
    "folk": "Waltz",
    "waltz": "Waltz",
    "march": "March",
    "jazz": "Jazz Swing",
    "jazz swing": "Jazz Swing",
    "swing": "Jazz Swing",
    "edm": "EDM",
    "electronic": "EDM",
    "sparse": "Sparse_Arpeggio",
    "sparse arpeggio": "Sparse_Arpeggio",
    "sparse_arpeggio": "Sparse_Arpeggio",
    "metal": "Metal",
}

_group_a_model = None
_melody_model = None
_melody_tokenizer = None
_melody_ready = False


def _normalize_style(s: str) -> str:
    s0 = (s or "").strip()
    if not s0:
        return "Pop Ballad"
    key = s0.lower().replace("_", " ").strip()
    mapped = _STYLE_ALIASES.get(key) or _STYLE_ALIASES.get(s0.lower().strip())
    if mapped and mapped in VALID_STYLES:
        return mapped
    for v in VALID_STYLES:
        if v.lower() == key:
            return v
    return "Pop Ballad"


def _is_midi_path(path: str) -> bool:
    _, ext = os.path.splitext((path or "").lower())
    return ext in [".mid", ".midi"]


def _safe_basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _try_import_music21():
    try:
        import music21 as m21  # type: ignore
        return m21
    except Exception:
        return None


def _quantize_midi_to_tokens(midi_path: str, limit_steps: int = 256) -> Tuple[List[str], List[int]]:
    m21 = _try_import_music21()
    if not m21:
        return [], []
    try:
        score = m21.converter.parse(midi_path)
        flat = score.flat
        notes = list(flat.notesAndRests)
        tokens: List[str] = []
        pos_list: List[int] = []
        step = 0
        for el in notes:
            if step >= limit_steps:
                break
            if getattr(el, "isRest", False):
                tokens.append("0")
                pos_list.append(step)
                step += 1
                continue
            if getattr(el, "isChord", False):
                try:
                    p = el.pitches[-1]
                    tokens.append(str(int(p.midi)))
                except Exception:
                    tokens.append("0")
                pos_list.append(step)
                step += 1
                continue
            try:
                tokens.append(str(int(el.pitch.midi)))
            except Exception:
                tokens.append("0")
            pos_list.append(step)
            step += 1
        return tokens, pos_list
    except Exception:
        return [], []


def _ensure_pkg(name: str):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


def _load_module_from_file(modname: str, filepath: str, transform_src=None):
    if modname in sys.modules:
        return sys.modules[modname]
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    if transform_src is not None:
        src = transform_src(src)
    m = types.ModuleType(modname)
    m.__file__ = filepath
    sys.modules[modname] = m
    exec(compile(src, filepath, "exec"), m.__dict__)
    return m


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _bootstrap_texture_render_B() -> bool:
    _ensure_pkg("src")
    _ensure_pkg("src.TextureRender_B")
    _ensure_pkg("TextureRender_B")

    data_fp = _find_first_existing([
        os.path.join(BASE_DIR, "src", "TextureRender_B", "data_logic_B.py"),
        os.path.join(BASE_DIR, "TextureRender_B", "data_logic_B.py"),
        os.path.join(BASE_DIR, "data_logic_B.py"),
    ])
    analysis_fp = _find_first_existing([
        os.path.join(BASE_DIR, "src", "TextureRender_B", "analysis_B.py"),
        os.path.join(BASE_DIR, "TextureRender_B", "analysis_B.py"),
        os.path.join(BASE_DIR, "analysis_B.py"),
    ])

    ok = True
    try:
        if data_fp:
            m_data = _load_module_from_file("_texB_data_logic", data_fp)
            sys.modules["src.TextureRender_B.data_logic_B"] = m_data
            sys.modules["TextureRender_B.data_logic_B"] = m_data
            sys.modules["data_logic_B"] = m_data
        else:
            ok = False
    except Exception:
        ok = False

    try:
        if analysis_fp:
            m_an = _load_module_from_file("_texB_analysis", analysis_fp)
            sys.modules["src.TextureRender_B.analysis_B"] = m_an
            sys.modules["TextureRender_B.analysis_B"] = m_an
            sys.modules["analysis_B"] = m_an
        else:
            ok = False
    except Exception:
        ok = False

    return ok


def _load_decision_logic_B_sanitized() -> Optional[types.ModuleType]:
    fp = _find_first_existing([
        os.path.join(BASE_DIR, "src", "TextureRender_B", "decision_logic_B.py"),
        os.path.join(BASE_DIR, "TextureRender_B", "decision_logic_B.py"),
        os.path.join(BASE_DIR, "decision_logic_B.py"),
    ])
    if not fp:
        return None

    def _strip_testish(src: str) -> str:
        out_lines = []
        for line in src.splitlines(True):
            if "if project_root not in sys.path" in line:
                continue
            if "sys.path.insert" in line:
                continue
            if "The root directory has been mounted." in line:
                continue
            if "print(f\"🔧" in line or "print(f'🔧" in line:
                continue
            out_lines.append(line)
        return "".join(out_lines)

    try:
        return _load_module_from_file("_texB_decision_logic", fp, transform_src=_strip_testish)
    except Exception:
        return None


def predict_harmony_from_midi(midi_path: str) -> List[str]:
    logger = logging.getLogger("GroupA")
    global _group_a_model
    try:
        if _group_a_model is None:
            try:
                from src.ChordGenerator_A.inference import AIComposer  # type: ignore
                _group_a_model = AIComposer()
                logger.info("Group A model loaded (src.ChordGenerator_A.inference).")
            except Exception:
                try:
                    from inference import AIComposer  # type: ignore
                    _group_a_model = AIComposer()
                    logger.info("Group A model loaded (inference.py).")
                except Exception as exc:
                    logger.warning("Group A model not available: %s", exc)
                    _group_a_model = None

        if _group_a_model is None:
            return ["C", "G", "Am", "F"] * 8

        tokens, pos_list = _quantize_midi_to_tokens(midi_path)
        if not tokens:
            return ["C", "G", "Am", "F"] * 8

        pred = getattr(_group_a_model, "predict", None)
        if not callable(pred):
            return ["C", "G", "Am", "F"] * 8

        try:
            sig = inspect.signature(pred)
            if "pos_list" in sig.parameters:
                chords = pred(tokens, pos_list)
            else:
                chords = pred(tokens)
        except Exception:
            chords = pred(tokens)

        if isinstance(chords, (list, tuple)) and chords:
            return [str(x) for x in chords]
    except Exception as exc:
        logger.warning("Harmony prediction failed: %s", exc)

    return ["C", "G", "Am", "F"] * 8


def detect_style_from_midi(midi_path: str) -> str:
    logger = logging.getLogger("Style")
    try:
        try:
            from src.StyleDetector_A.inference import detect_style  # type: ignore
        except Exception:
            from detect_style import detect_style  # type: ignore

        try:
            sig = inspect.signature(detect_style)
            kwargs = {}
            if "midi_path" in sig.parameters:
                kwargs["midi_path"] = midi_path
            elif "path" in sig.parameters:
                kwargs["path"] = midi_path
            if "pos_list" in sig.parameters:
                _, pos_list = _quantize_midi_to_tokens(midi_path)
                kwargs["pos_list"] = pos_list
            s = detect_style(**kwargs) if kwargs else detect_style(midi_path)
        except Exception:
            s = detect_style(midi_path)

        s2 = _normalize_style(str(s))
        if s2 in VALID_STYLES and s2 != "Auto":
            return s2
    except Exception as exc:
        logger.info("Style detector not available: %s", exc)
    return "Pop Ballad"


def _call_renderer_B(melody_midi_path: str, chords: List[str], style: str, mode: str, tempo: Optional[str]):
    logger = logging.getLogger("GroupB")
    try:
        _bootstrap_texture_render_B()

        m = _load_decision_logic_B_sanitized()
        render_fn = getattr(m, "render_accompaniment_from_raw_inputs", None) if m else None

        if not render_fn:
            for mod in ("src.TextureRender_B.decision_logic_B", "TextureRender_B.decision_logic_B", "decision_logic_B"):
                try:
                    mm = __import__(mod, fromlist=["render_accompaniment_from_raw_inputs"])
                    render_fn = getattr(mm, "render_accompaniment_from_raw_inputs", None)
                    if render_fn:
                        break
                except Exception:
                    continue

        if not render_fn:
            return None

        try:
            sig = inspect.signature(render_fn)
        except Exception:
            return render_fn(melody_midi_path, chords, style)

        args = []
        kwargs = {}

        params = list(sig.parameters.keys())
        if len(params) >= 1:
            args.append(melody_midi_path)
        if len(params) >= 2:
            args.append(chords)
        if len(params) >= 3:
            args.append(style)

        if "mode" in sig.parameters:
            kwargs["mode"] = mode
        if "tempo" in sig.parameters:
            kwargs["tempo"] = tempo

        return render_fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("Group B renderer failed: %s", exc)
        return None


def render_music(
    melody_midi_path: str,
    chord_sequence: List[str],
    style: str,
    mode: str = "melody",
    tempo: Optional[str] = None,
) -> str:
    logger = logging.getLogger("Render")
    style = _normalize_style(style)
    if style == "Auto":
        style = "Pop Ballad"

    out_base = f"{_safe_basename_noext(melody_midi_path)}_{style.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    midi_out = os.path.join(OUTPUT_DIR, out_base + ".mid")

    if not os.path.exists(melody_midi_path) or not _is_midi_path(melody_midi_path):
        raise FileNotFoundError("Input MIDI not found or invalid.")

    m21 = _try_import_music21()
    accompaniment_part = _call_renderer_B(melody_midi_path, chord_sequence, style, mode, tempo)

    if not m21 or accompaniment_part is None:
        shutil.copy2(melody_midi_path, midi_out)
        return midi_out

    score = m21.converter.parse(melody_midi_path)
    if getattr(score, "parts", None):
        base_score = score
    else:
        base_score = m21.stream.Score()
        base_score.insert(0, score)

    if tempo:
        try:
            bpm = int("".join([c for c in str(tempo) if c.isdigit()]) or "0")
            if bpm > 0:
                base_score.insert(0, m21.tempo.MetronomeMark(number=bpm))
        except Exception:
            pass

    try:
        base_score.append(accompaniment_part)
    except Exception:
        try:
            base_score.insert(0, accompaniment_part)
        except Exception:
            shutil.copy2(melody_midi_path, midi_out)
            return midi_out

    base_score.write("midi", fp=midi_out)
    logger.info("Rendered MIDI: %s", midi_out)
    return midi_out


_SOUND_FONT_CACHE = None

def _resolve_ffmpeg() -> Optional[str]:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    la = os.environ.get("LOCALAPPDATA") or ""
    wa = os.path.join(la, "Microsoft", "WindowsApps", "ffmpeg.exe")
    if os.path.exists(wa):
        return wa
    return None

def _resolve_fluidsynth() -> Optional[str]:
    fs = shutil.which("fluidsynth")
    if fs:
        return fs
    la = os.environ.get("LOCALAPPDATA") or ""
    wa = os.path.join(la, "Microsoft", "WindowsApps", "fluidsynth.exe")
    if os.path.exists(wa):
        return wa
    return None

def _find_soundfont() -> Optional[str]:
    global _SOUND_FONT_CACHE
    if _SOUND_FONT_CACHE is not None:
        return _SOUND_FONT_CACHE

    cands = [
        os.path.join(BASE_DIR, "soundfont.sf2"),
        os.path.join(BASE_DIR, "assets", "soundfont.sf2"),
        os.path.join(BASE_DIR, "assets", "FluidR3_GM.sf2"),
        os.path.join(BASE_DIR, "assets", "GeneralUserGS.sf2"),
    ]
    for p in cands:
        if p and os.path.exists(p):
            _SOUND_FONT_CACHE = p
            return p

    assets_dir = os.path.join(BASE_DIR, "assets")
    if os.path.isdir(assets_dir):
        for root, _, files in os.walk(assets_dir):
            for f in files:
                if f.lower().endswith(".sf2"):
                    p = os.path.join(root, f)
                    _SOUND_FONT_CACHE = p
                    return p

    _SOUND_FONT_CACHE = None
    return None

def midi_to_mp3(midi_path: str) -> Optional[str]:
    logger = logging.getLogger("ConvertMP3")
    if not midi_path or not os.path.exists(midi_path):
        return None

    base = _safe_basename_noext(midi_path)
    mp3_out = os.path.join(OUTPUT_DIR, base + ".mp3")
    if os.path.exists(mp3_out) and os.path.getsize(mp3_out) > 0:
        return mp3_out

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        logger.info("ffmpeg not found (python PATH).")
        return None

    fluidsynth = _resolve_fluidsynth()
    sf2 = _find_soundfont()


    if fluidsynth and sf2 and os.path.exists(sf2):
        wav_tmp = os.path.join(OUTPUT_DIR, base + ".__tmp__.wav")
        try:
            cmd1 = [fluidsynth, "-ni", sf2, midi_path, "-F", wav_tmp, "-r", "44100"]
            subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            cmd2 = [ffmpeg, "-y", "-i", wav_tmp, "-codec:a", "libmp3lame", "-q:a", "2", mp3_out]
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(mp3_out) and os.path.getsize(mp3_out) > 0:
                return mp3_out
        except subprocess.CalledProcessError as e:
            try:
                logger.info("fluidsynth/ffmpeg failed: %s", (e.stderr or b"").decode("utf-8", "ignore"))
            except Exception:
                logger.info("fluidsynth/ffmpeg failed.")
        finally:
            try:
                if os.path.exists(wav_tmp):
                    os.remove(wav_tmp)
            except Exception:
                pass


    try:
        cmd = [ffmpeg, "-y", "-i", midi_path, mp3_out]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(mp3_out) and os.path.getsize(mp3_out) > 0:
            return mp3_out
    except subprocess.CalledProcessError as e:
        try:
            logger.info("ffmpeg direct failed: %s", (e.stderr or b"").decode("utf-8", "ignore"))
        except Exception:
            logger.info("ffmpeg direct failed.")
    return None



def midi_to_musicxml(midi_path: str) -> Optional[str]:
    if not midi_path or not os.path.exists(midi_path):
        return None
    base = _safe_basename_noext(midi_path)
    out = os.path.join(OUTPUT_DIR, base + ".musicxml")
    if os.path.exists(out):
        return out
    m21 = _try_import_music21()
    if not m21:
        return None
    try:
        score = m21.converter.parse(midi_path)
        score.write("musicxml", fp=out)
        if os.path.exists(out):
            return out
    except Exception:
        return None
    return None


def _ensure_melody_model_loaded() -> bool:
    global _melody_model, _melody_tokenizer, _melody_ready
    if _melody_ready:
        return True

    logger = logging.getLogger("MelodyLoad")

    def _first_exist(cands):
        for p in cands:
            if p and os.path.exists(p):
                return p
        return None

    try:
        import torch  # type: ignore
    except Exception as e:
        logger.error("torch not available: %s", e)
        return False


    root2 = os.path.abspath(os.path.join(BASE_DIR, ".."))

    vocab_candidates = [
        os.path.join(BASE_DIR, "data", "processed", "Melody_vocab.json"),
        os.path.join(root2,   "data", "processed", "Melody_vocab.json"),
        os.path.join(BASE_DIR, "src", "MelodyGenerator_A", "Melody_vocab.json"),
        os.path.join(root2,   "src", "MelodyGenerator_A", "Melody_vocab.json"),
    ]
    ckpt_candidates = [
        os.path.join(BASE_DIR, "src", "MelodyGenerator_A", "Melody_ckpt.pt"),
        os.path.join(root2,   "src", "MelodyGenerator_A", "Melody_ckpt.pt"),
        os.path.join(BASE_DIR, "src", "MelodyGenerator_A", "Melody_ckpt.pth"),
        os.path.join(root2,   "src", "MelodyGenerator_A", "Melody_ckpt.pth"),
    ]

    vocab_path = _first_exist(vocab_candidates)
    ckpt_path = _first_exist(ckpt_candidates)

    if not vocab_path:
        logger.error("Melody vocab not found. Tried: %s", vocab_candidates)
        return False
    if not ckpt_path:
        logger.error("Melody checkpoint not found. Tried: %s", ckpt_candidates)
        return False


    try:
        from src.MelodyGenerator_A.Melody_tokenizer import MelodyTokenizer  # type: ignore
        from src.MelodyGenerator_A.Melody_model import MelodyGPT, GPTConfig  # type: ignore
    except Exception:
        try:
            from MelodyGenerator_A.Melody_tokenizer import MelodyTokenizer  # type: ignore
            from MelodyGenerator_A.Melody_model import MelodyGPT, GPTConfig  # type: ignore
        except Exception as e:
            logger.error("MelodyGenerator_A imports failed: %s", e)
            return False

    try:
        tok = MelodyTokenizer()
        tok.load_vocab(vocab_path)

        tok = MelodyTokenizer()
        tok.load_vocab(vocab_path)

        vocab_size = getattr(tok, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(tok, "n_vocab", None)
        if vocab_size is None:
            for attr in ("vocab", "stoi", "itos", "token_to_id", "id_to_token"):
                obj = getattr(tok, attr, None)
                if obj is None:
                    continue
                try:
                    vocab_size = len(obj)
                    break
                except Exception:
                    pass
        if vocab_size is None:
            raise RuntimeError("Tokenizer vocab size unavailable")

        tok = MelodyTokenizer()
        tok.load_vocab(vocab_path)

        vocab_size = getattr(tok, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(tok, "n_vocab", None)
        if vocab_size is None:
            for attr in ("vocab", "stoi", "itos", "token_to_id", "id_to_token"):
                obj = getattr(tok, attr, None)
                if obj is None:
                    continue
                try:
                    vocab_size = len(obj)
                    break
                except Exception:
                    pass
        if vocab_size is None:
            raise RuntimeError("Tokenizer vocab size unavailable")

        try:
            cfg = GPTConfig(
                block_size=256,
                vocab_size=int(vocab_size),
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.0,
                bias=False,
            )
        except TypeError:
            cfg = GPTConfig(
                block_size=256,
                vocab_size=int(vocab_size),
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.0,
            )

        model = MelodyGPT(cfg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj = torch.load(ckpt_path, map_location=device)


        if isinstance(obj, dict):
            if "model_state_dict" in obj:
                state = obj["model_state_dict"]
            elif "state_dict" in obj:
                state = obj["state_dict"]
            else:
                state = obj
        else:
            state = obj

        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        _melody_model = model
        _melody_tokenizer = tok
        _melody_ready = True

        logger.info("Melody model loaded. vocab=%s ckpt=%s device=%s", vocab_path, ckpt_path, device)
        return True

    except Exception as e:
        logger.error("Melody load failed: %s", e, exc_info=True)
        return False


def generate_melody_midi(
    key: str = "C",
    meter: str = "4/4",
    seed: Optional[int] = None,
    max_tokens: int = 500,
    temp: float = 0.8,
) -> str:
    logger = logging.getLogger("MelodyGen")

    if seed is not None:
        try:
            random.seed(int(seed))
        except Exception:
            pass

    if not _ensure_melody_model_loaded():
        raise RuntimeError("Melody model not available.")

    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torch not available: {e}")

    def _extract_first_song(full_text: str) -> str:
        lines = full_text.split("\n")
        valid = []
        seen_k = False
        for i, line in enumerate(lines):
            line = (line or "").strip()
            if not line:
                continue
            if i > 0 and re.match(r"^\d+$", line):
                break
            is_header = re.match(r"^[A-Z]:", line) is not None
            if seen_k and is_header:
                break
            valid.append(line)
            if line.startswith("K:"):
                seen_k = True
        return "\n".join(valid)

    def _write_fallback_midi(path: str) -> None:
        m21 = _try_import_music21()
        if m21:
            s = m21.stream.Stream()
            try:
                s.append(m21.meter.TimeSignature(meter))
            except Exception:
                try:
                    s.append(m21.meter.TimeSignature("4/4"))
                except Exception:
                    pass
            try:
                s.append(m21.key.Key(key))
            except Exception:
                pass
            notes = [60, 62, 64, 65, 67, 69, 71, 72]
            for n in notes:
                nn = m21.note.Note(n)
                nn.quarterLength = 1
                s.append(nn)
            s.write("midi", fp=path)
            return

        try:
            import mido  # type: ignore
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.Message("program_change", program=0, time=0))
            notes = [60, 62, 64, 65, 67, 69, 71, 72]
            dur = 480
            for n in notes:
                track.append(mido.Message("note_on", note=n, velocity=64, time=0))
                track.append(mido.Message("note_off", note=n, velocity=64, time=dur))
            mid.save(path)
            return
        except Exception:
            raise RuntimeError("No midi writer available (music21/mido missing).")

    out_base = f"melody_{key}_{meter.replace('/', '-')}_{uuid.uuid4().hex[:8]}"
    midi_out = os.path.join(OUTPUT_DIR, out_base + ".mid")

    try:
        model = _melody_model
        tok = _melody_tokenizer
        if model is None or tok is None:
            raise RuntimeError("Melody model/tokenizer not ready.")

        device = None
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        try:
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

        seed_text = f"M:{meter}\nK:{key}\n"
        if not hasattr(tok, "encode") or not hasattr(tok, "decode"):
            raise RuntimeError("Tokenizer missing encode/decode.")

        start_ids = tok.encode(seed_text)
        x = torch.tensor([start_ids], dtype=torch.long, device=device)

        gen_fn = getattr(model, "generate", None)
        if not callable(gen_fn):
            raise RuntimeError("Model missing generate().")

        with torch.no_grad():
            y = gen_fn(x, max_new_tokens=max_tokens, temperature=temp, top_k=40)

        ids = y[0].tolist() if hasattr(y, "__getitem__") else list(y)
        full_text = tok.decode(ids)

        abc = _extract_first_song(full_text)
        if "X:" not in abc:
            abc = "X:1\n" + abc

        m21 = _try_import_music21()
        if m21:
            try:
                s = m21.converter.parse(abc, format="abc")
                if isinstance(s, m21.stream.Opus):
                    s = s[0]
                s.write("midi", fp=midi_out)
                if os.path.exists(midi_out):
                    logger.info("Melody MIDI: %s", midi_out)
                    return midi_out
            except Exception:
                pass

        _write_fallback_midi(midi_out)
        logger.info("Melody MIDI (fallback): %s", midi_out)
        return midi_out

    except Exception as e:
        logger.error("Melody generation failed: %s", e, exc_info=True)
        try:
            _write_fallback_midi(midi_out)
            logger.info("Melody MIDI (fallback after error): %s", midi_out)
            return midi_out
        except Exception:
            raise



def generate_song(
    uploaded_midi_path: str,
    selected_style: str = "Auto",
    mode: str = "chord",
    tempo: Optional[str] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger("Pipeline")
    result: Dict[str, Any] = {
        "success": False,
        "message": "",
        "midi_path": "",
        "mp3_path": "",
        "musicxml_path": "",
        "detected_style": "",
        "final_style": "",
        "mode": mode,
        "tempo": tempo or "",
        "chord_preview": [],
    }

    if not uploaded_midi_path or not os.path.exists(uploaded_midi_path):
        result["message"] = "Input file not found."
        return result
    if not _is_midi_path(uploaded_midi_path):
        result["message"] = "Only MIDI files (.mid/.midi) are supported."
        return result

    try:
        detected = detect_style_from_midi(uploaded_midi_path)
        result["detected_style"] = detected

        sel = _normalize_style(selected_style)
        if sel == "Auto":
            final_style = detected
        else:
            final_style = sel

        if final_style not in VALID_STYLES or final_style == "Auto":
            final_style = "Pop Ballad"
        if final_style == "Metal":
            final_style = "EDM"

        result["final_style"] = final_style

        chords = predict_harmony_from_midi(uploaded_midi_path)
        result["chord_preview"] = chords[:32]

        midi_out = render_music(
            melody_midi_path=uploaded_midi_path,
            chord_sequence=chords,
            style=final_style,
            mode=mode,
            tempo=tempo,
        )
        result["midi_path"] = midi_out

        mp3 = midi_to_mp3(midi_out)
        if mp3:
            result["mp3_path"] = mp3

        mx = midi_to_musicxml(midi_out)
        if mx:
            result["musicxml_path"] = mx

        result["success"] = True
        result["message"] = "Generation successful! Please click to play or download."
        logger.info("Generation OK: %s", midi_out)
        return result
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        result["success"] = False
        result["message"] = f"Generation failed: {exc}"
        return result
