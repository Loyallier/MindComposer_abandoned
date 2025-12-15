# 保存为 check_midi_length.py
import music21
import os
from src import path

midi_path = os.path.join("test_melody.mid") # 确保路径对

print(f"🧐 正在诊断: {midi_path}")
s = music21.converter.parse(midi_path)
length = s.duration.quarterLength
print(f"⏱️ MIDI 总长度 (拍数): {length}")
print(f"🎼 估算小节数 (按4/4拍): {length / 4:.2f}")

# 看看是不是有超长音
notes = list(s.recurse().notes)
print(f"🎹 音符总数: {len(notes)}")
if notes:
    print(f"   - 第一个音位置: {notes[0].offset}")
    print(f"   - 最后一个音位置: {notes[-1].offset}")
    print(f"   - 最后一个音长度: {notes[-1].duration.quarterLength}")