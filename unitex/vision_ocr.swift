// Apple Vision text recognition backend for unitex/ocr.py (macOS only).
//
// Build (ocr.py does this on first use):
//   /usr/bin/swiftc -O unitex/vision_ocr.swift -o ~/.cache/unitex/vision_ocr
//
// Usage:
//   vision_ocr [--langs en-US,fr-FR] [--fast] [--correction] [--min-height F] img1 [img2 ...]
//
// Prints one JSON object per input image, one per line, in input order:
//   {"path": ..., "w": W, "h": H, "items": [{"text": ..., "conf": ..., "quad_norm": [[x, y] x 4]}]}
// or {"path": ..., "error": ...} when the image cannot be read or recognition fails.
//
// quad_norm is in Vision's normalized image space: origin at the BOTTOM-left, y pointing up,
// corners ordered top-left, top-right, bottom-right, bottom-left in the text's reading frame.
// ocr.py converts it to top-left-origin pixels: x_px = x * W, y_px = (1 - y) * H.

import Foundation
import ImageIO
import Vision

struct Options {
    var langs: [String] = []
    var fast = false
    var correction = false
    var minHeight: Float = -1        // < 0 keeps Vision's default
    var paths: [String] = []
}

func parseArgs() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "--langs":
            if !args.isEmpty { o.langs = args.removeFirst().components(separatedBy: ",").filter { !$0.isEmpty } }
        case "--fast":
            o.fast = true
        case "--correction":
            o.correction = true
        case "--min-height":
            if !args.isEmpty { o.minHeight = Float(args.removeFirst()) ?? -1 }
        default:
            o.paths.append(a)
        }
    }
    return o
}

// ImageIO keeps the exact pixel size (NSImage would go through point sizes and DPI metadata).
func loadCGImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path) as CFURL
    guard let src = CGImageSourceCreateWithURL(url, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func emit(_ obj: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: obj, options: []),
       let s = String(data: d, encoding: .utf8) {
        print(s)
    } else {
        print("{\"error\": \"json serialization failed\"}")
    }
    fflush(stdout)
}

func pt(_ p: CGPoint) -> [Double] { [Double(p.x), Double(p.y)] }

let opts = parseArgs()
if opts.paths.isEmpty {
    FileHandle.standardError.write("usage: vision_ocr [--langs a,b] [--fast] [--correction] [--min-height F] img...\n".data(using: .utf8)!)
    exit(2)
}

for path in opts.paths {
    guard let cg = loadCGImage(path) else {
        emit(["path": path, "error": "cannot load image"])
        continue
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = opts.fast ? .fast : .accurate
    req.usesLanguageCorrection = opts.correction
    if !opts.langs.isEmpty { req.recognitionLanguages = opts.langs }
    if opts.minHeight >= 0 { req.minimumTextHeight = opts.minHeight }
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([req])
    } catch {
        emit(["path": path, "w": cg.width, "h": cg.height, "error": "\(error)"])
        continue
    }
    var items: [[String: Any]] = []
    for obs in req.results ?? [] {
        guard let cand = obs.topCandidates(1).first else { continue }
        items.append([
            "text": cand.string,
            "conf": Double(cand.confidence),
            "quad_norm": [pt(obs.topLeft), pt(obs.topRight), pt(obs.bottomRight), pt(obs.bottomLeft)],
        ])
    }
    emit(["path": path, "w": cg.width, "h": cg.height, "items": items])
}
