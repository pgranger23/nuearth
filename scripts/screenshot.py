"""Headless-Firefox screenshots of the running app, via Firefox's built-in
Marionette protocol (no geckodriver needed).

usage: python scripts/screenshot.py URL OUT.png [--size 1600x1100] [--js "code"] [--wait-ready]
"""
import argparse, base64, json, os, shutil, socket, subprocess, tempfile, time


class Marionette:
    def __init__(self, port):
        for _ in range(100):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=60)
                break
            except OSError:
                time.sleep(0.2)
        self.buf = b""
        self.msg_id = 0
        self._read()  # handshake

    def _read(self):
        while b":" not in self.buf:
            self.buf += self.s.recv(65536)
        n, rest = self.buf.split(b":", 1)
        n = int(n)
        while len(rest) < n:
            rest += self.s.recv(1 << 20)
        self.buf = rest[n:]
        return json.loads(rest[:n])

    def cmd(self, name, params=None):
        self.msg_id += 1
        data = json.dumps([0, self.msg_id, name, params or {}]).encode()
        self.s.sendall(str(len(data)).encode() + b":" + data)
        while True:
            r = self._read()
            if r[0] == 1 and r[1] == self.msg_id:
                if r[2]:
                    raise RuntimeError(r[2])
                return r[3]

    def js(self, code):
        return self.cmd("WebDriver:ExecuteScript", {"script": code, "args": []}).get("value")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--size", default="1600x1100")
    ap.add_argument("--js", action="append", default=[])
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    prof = tempfile.mkdtemp(prefix="ffprof-")
    port = 28280 + os.getpid() % 500
    with open(os.path.join(prof, "user.js"), "w") as f:
        f.write(f'user_pref("marionette.port", {port});\n')
        f.write('user_pref("browser.shell.checkDefaultBrowser", false);\n')
        f.write('user_pref("datareporting.policy.dataSubmissionEnabled", false);\n')
    ff = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", prof,
                           f"--width={w}", f"--height={h}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        m = Marionette(port)
        m.cmd("WebDriver:NewSession", {"capabilities": {}})
        m.cmd("WebDriver:SetWindowRect", {"width": w, "height": h})
        m.cmd("WebDriver:Navigate", {"url": a.url})

        def wait_ready():
            for _ in range(240):
                st = m.js("const e = document.getElementById('status'); return e ? e.textContent : '';")
                if "ready" in (st or ""):
                    return st
                time.sleep(0.25)
            return st
        st = wait_ready()
        for code in a.js:
            m.js(code)
            time.sleep(1.0)
            st = wait_ready()
        time.sleep(1.5)
        print("status:", st, "| js errors:", m.js("return window.__errs || []"))
        if a.full:
            full_h = m.js("return document.documentElement.scrollHeight")
            m.cmd("WebDriver:SetWindowRect", {"width": w, "height": int(full_h) + 90})
            time.sleep(1.5)
        png = m.cmd("WebDriver:TakeScreenshot", {"full": bool(a.full)})["value"]
        with open(a.out, "wb") as f:
            f.write(base64.b64decode(png))
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
