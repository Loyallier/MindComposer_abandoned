import music21
import os
import glob
import re
from tqdm import tqdm

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# 定义特定的文件名列表，或者扫描所有 txt
TARGET_FILES = ["reelsh-l.txt", "reelsd-g.txt"]
# 如果你想清洗所有文件，可以使用: TARGET_FILES = glob.glob(os.path.join(RAW_DIR, "*.txt"))


def split_abc_content(content):
    """
    将 ABC 文件内容拆分为独立的曲目列表。
    逻辑：ABC 文件总是以 X: 编号开始作为一首新曲目的标志。
    """
    lines = content.splitlines()
    songs = []
    current_song_lines = []

    for line in lines:
        # 移除可能存在的 标记干扰 (如果它是单独一行)
        if line.strip().startswith("[source"):
            # 这种标记往往不是合法的 ABC 语法，可能会被前面的工具注入
            # 我们尝试清洗它，或者如果它在 X: 前面，我们忽略它
            clean_line = re.sub(r"\\", "", line).strip()
            if not clean_line:
                continue  # 如果清洗后为空，跳过
            line = clean_line

        # 检测 X: 标记作为新曲目的开始
        # 标准 ABC 格式：X: 1 (行首)
        if re.match(r"^X:\s*\d+", line.strip()):
            if current_song_lines:
                songs.append("\n".join(current_song_lines))
            current_song_lines = [line]
        else:
            current_song_lines.append(line)

    # 添加最后一首
    if current_song_lines:
        songs.append("\n".join(current_song_lines))

    return songs


def clean_and_save():
    print("🧹 开始执行外科手术式清洗...")
    print(f"📂 目标目录: {RAW_DIR}")

    total_removed = 0
    total_kept = 0

    # 获取所有需要处理的文件路径
    files_to_process = [os.path.join(RAW_DIR, f) for f in TARGET_FILES]

    # 也可以自动扫描所有 txt，防止遗漏
    # files_to_process = glob.glob(os.path.join(RAW_DIR, "*.txt"))

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件未找到: {file_path}")
            continue

        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}...")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 1. 拆分曲目
        raw_songs = split_abc_content(content)
        valid_songs = []
        file_removed_count = 0

        # 2. 逐一验证
        for song_str in tqdm(raw_songs, desc=f"Scanning {filename}"):
            if not song_str.strip():
                continue

            try:
                # === 核心逻辑：尝试解析 ===
                # 如果这一步报错，说明这首歌有致命语法错误
                # 我们不需要知道错在哪，我们只需要知道它“坏了”
                s = music21.converter.parse(song_str, format="abc")

                # 额外的双重检查：是否存在未闭合的方括号（针对你提到的具体错误）
                # 虽然 parse 通常会捕获，但显式检查更安全
                if song_str.count("[") != song_str.count("]"):
                    raise ValueError("Unbalanced brackets mismatch")

                valid_songs.append(song_str)

            except Exception as e:
                # print(f"   ❌ Drop song X:{get_x_number(song_str)} - Reason: {str(e)[:50]}...")
                file_removed_count += 1

        # 3. 覆写文件 (保存清洗后的版本)
        if valid_songs:
            # 用双换行符连接，确保格式清晰
            new_content = "\n\n".join(valid_songs)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(
                f"   ✅ {filename} 处理完成: 保留 {len(valid_songs)} 首, 删除 {file_removed_count} 首坏曲目。"
            )
            total_kept += len(valid_songs)
            total_removed += file_removed_count
        else:
            print(f"   ⚠️ {filename} 处理后为空！未写入。")

    print("\n" + "=" * 40)
    print(f"🎉 清洗总结:")
    print(f"   - 总保留曲目: {total_kept}")
    print(f"   - 总剔除坏曲: {total_removed}")
    print("   - 原始文件已被原地更新为清洗后的版本。")
    print("=" * 40)


def get_x_number(song_str):
    """辅助函数：提取 X 编号用于日志"""
    match = re.search(r"^X:\s*(\d+)", song_str, re.MULTILINE)
    return match.group(1) if match else "?"


if __name__ == "__main__":
    clean_and_save()
