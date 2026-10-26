"""Record a zenith sweep (oscillogram + Earth journey + flavor along the path) as a GIF.
usage: python scripts/make_gif.py URL OUT.gif [--E 6]"""
import argparse, base64, io, os, shutil, subprocess, sys, tempfile, time
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from screenshot import Marionette


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--E", type=float, default=6.0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--setup", default="")
    a = ap.parse_args()
    prof = tempfile.mkdtemp(prefix="ffprof-")
    port = 28800 + os.getpid() % 500
    with open(os.path.join(prof, "user.js"), "w") as f:
        f.write(f'user_pref("marionette.port", {port});\n')
    ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", prof],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        m = Marionette(port)
        m.cmd("WebDriver:NewSession", {"capabilities": {}})
        m.cmd("WebDriver:SetWindowRect", {"width": 1600, "height": 2200})
        m.cmd("WebDriver:Navigate", {"url": a.url})
        for _ in range(240):
            if "ready" in (m.js("return document.getElementById('status').textContent") or ""):
                break
            time.sleep(0.25)
        if a.setup:
            m.js(a.setup); time.sleep(3)

        def el(css):
            return list(m.cmd("WebDriver:FindElement", {"using": "css selector", "value": css})["value"].values())[0]
        cards = [el(".center > .card"), el(".right > .card:nth-child(1)"), el(".right > .card:nth-child(2)")]
        frames = []
        n = a.frames
        czs = [-0.05 - 0.95 * i / (n - 1) for i in range(n)]
        for cz in czs:
            m.js(f"nuearth.select({a.E}, {cz});")
            for _ in range(80):
                if abs((m.js("return nuearth.path ? nuearth.path.cz : 9") or 9) - cz) < 1e-9:
                    break
                time.sleep(0.05)
            time.sleep(0.15)
            imgs = [Image.open(io.BytesIO(base64.b64decode(
                m.cmd("WebDriver:TakeScreenshot", {"id": c})["value"]))).convert("RGB") for c in cards]
            osc, earth, path = imgs
            right_w = max(earth.width, path.width)
            H = max(osc.height, earth.height + path.height)
            frame = Image.new("RGB", (osc.width + right_w + 12, H), (249, 249, 247))
            frame.paste(osc, (0, 0)); frame.paste(earth, (osc.width + 12, 0))
            frame.paste(path, (osc.width + 12, earth.height))
            frame = frame.resize((int(frame.width * 0.62), int(frame.height * 0.62)), Image.LANCZOS)
            frames.append(frame.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
        frames = frames + frames[-2:0:-1]
        durations = [140] * len(frames)
        durations[n - 1] = 900
        frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        print("frames", len(frames), "size", frames[0].size, os.path.getsize(a.out) // 1024, "kB")
        m.cmd("WebDriver:DeleteSession")
    finally:
        ff.terminate()
        try:
            ff.wait(10)
        except subprocess.TimeoutExpired:
            ff.kill()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
