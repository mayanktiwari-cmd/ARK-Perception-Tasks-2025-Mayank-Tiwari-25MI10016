import cv2
import numpy as np
import os
import sys
videos = ["1.mp4", "2.mp4", "3.mp4"]
os.makedirs("output", exist_ok=True)
scale      = 0.5
min_votes  = 20
max_pts    = 1000
angle_step = 1
def get_edges(gray):
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    _, out = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
    return out
def find_lines(edges):
    h, w   = edges.shape
    diag   = int(np.sqrt(h**2 + w**2))
    angles = np.deg2rad(np.arange(0, 180, angle_step))
    ca     = np.cos(angles)
    sa     = np.sin(angles)
    acc    = np.zeros((2 * diag, len(angles)), dtype=np.int32)
    pts = np.argwhere(edges > 0)
    if len(pts) == 0:
        return []
    if len(pts) > max_pts:
        pts = pts[np.random.choice(len(pts), max_pts, replace=False)]
    ys = pts[:, 0].astype(np.float32)
    xs = pts[:, 1].astype(np.float32)
    rhos = (xs[:, None] * ca[None, :] + ys[:, None] * sa[None, :]).astype(np.int32)
    rhos = np.clip(rhos + diag, 0, 2 * diag - 1)
    aidx = np.tile(np.arange(len(angles)), (len(pts), 1))
    np.add.at(acc, (rhos.ravel(), aidx.ravel()), 1)
    result = []
    tmp    = acc.copy()
    for _ in range(15):
        flat   = np.argmax(tmp)
        ri, ai = np.unravel_index(flat, tmp.shape)
        v      = tmp[ri, ai]
        if v < min_votes:
            break
        result.append((ri - diag, angles[ai], int(v)))
        tmp[max(0, ri-15):ri+15, max(0, ai-15):ai+15] = 0
    return sorted(result, key=lambda x: x[2], reverse=True)
def to_points(rho, theta):
    a  = np.cos(theta)
    b  = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    return (int(x0 + 2000*(-b)), int(y0 + 2000*a)), \
           (int(x0 - 2000*(-b)), int(y0 - 2000*a))
def draw_axis(frame, lines, sc=1.0):
    out = frame.copy()
    if len(lines) < 2:
        cv2.putText(out, "no lines", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    r1, t1, _ = lines[0]
    r2, t2, _ = lines[1]
    if sc != 1.0:
        r1 /= sc
        r2 /= sc
    cv2.line(out, *to_points((r1+r2)/2, (t1+t2)/2), (0, 255, 0), 3)
    return out
def run(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cant open {video_path}")
        return
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pw    = int(w * scale)
    ph    = int(h * scale)
    print(f"{video_path}: {w}x{h} {fps}fps {total} frames")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    sub    = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    sk     = np.ones((3, 3), np.uint8)
    lk     = np.ones((5, 5), np.uint8)
    n      = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        small   = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        fg      = sub.apply(small)
        cleaned = cv2.morphologyEx(fg,      cv2.MORPH_OPEN,  sk)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, lk)
        edges   = get_edges(cleaned)
        lines   = find_lines(edges)
        result  = draw_axis(frame, lines, sc=scale)
        writer.write(result)
        if n % 3 == 0:
            cv2.imshow("original", frame)
            cv2.imshow("edges",    edges)
            cv2.imshow("result",   cv2.resize(result, (pw*2, ph*2)))
        if n % 20 == 0:
            print(f"  frame {n}/{total}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"done -> {out_path}")
def run_frames(folder, out_folder="output/frames_result"):
    os.makedirs(out_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print(f"no images in {folder}")
        return
    sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    sk  = np.ones((3, 3), np.uint8)
    lk  = np.ones((5, 5), np.uint8)
    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(folder, fname))
        if frame is None:
            continue
        fh, fw  = frame.shape[:2]
        pw, ph  = int(fw*scale), int(fh*scale)
        small   = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        fg      = sub.apply(small)
        cleaned = cv2.morphologyEx(fg,      cv2.MORPH_OPEN,  sk)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, lk)
        edges   = get_edges(cleaned)
        lines   = find_lines(edges)
        result  = draw_axis(frame, lines, sc=scale)
        cv2.imwrite(os.path.join(out_folder, f"result_{fname}"), result)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"saved to {out_folder}")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if os.path.isdir(p):
            run_frames(p)
        elif os.path.isfile(p):
            name = os.path.splitext(os.path.basename(p))[0]
            run(p, f"output/{name}_result.mp4")
        else:
            print(f"not found: {p}")
    else:
        done = 0
        for v in videos:
            if os.path.isfile(v):
                done += 1
                name = os.path.splitext(v)[0]
                print(f"\n[{done}/3] {v}")
                run(v, f"output/{name}_result.mp4")
            else:
                print(f"skipping {v} - not found")
        if done == 0:
            print("no videos found - rename your files to 1.mp4 2.mp4 3.mp4")
            print("or run: python medial_axis.py yourvideo.mp4")
        else:
            print(f"\nfinished {done} video(s) - check output/ folder")