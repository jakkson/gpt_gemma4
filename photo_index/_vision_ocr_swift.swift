// Vision OCR helper — invoked as a subprocess by photo_index.vision_ocr.
//
// macOS 26 (Tahoe) deprecated VNRecognizeTextRequest in favor of the
// async RecognizeTextRequest, and additionally requires images that fit
// CVPixelBuffer's YCbCr 4:2:0 layout (dimensions multiples of 16). This
// program normalizes the input image to those constraints before running
// Apple Vision OCR.
//
// Usage:
//   vision_ocr_swift <path> [<path> ...]
//
// Output: One block per input file, terminated with a single line "---END---"
// (or "---ERR---" on per-file failure). Recognized text lines are printed
// verbatim, one per line, before the terminator.

import Foundation
import Vision
import AppKit
import CoreGraphics

// Render the source image into a fresh BGRA CVPixelBuffer.
//
// macOS 26's Vision/RecognizeTextRequest internally tries to create a YCbCr
// 4:2:0 ('420f') pixel buffer when handed a CGImage, and that creation is
// brittle — empirically failing with "Failed to create CVPixelBuffer" on
// many real-world images. Bypassing it by handing Vision a pre-built BGRA
// CVPixelBuffer (which CoreImage / Vision will use directly) sidesteps the
// problem.
//
// We also downscale to keep the longest side <= maxSide and round dimensions
// to multiples of 16 — the alignment is harmless for BGRA but covers any
// downstream pipeline that re-derives a planar buffer from this one.
func renderToPixelBuffer(_ src: CGImage, maxSide: Int = 2048, align: Int = 16) -> CVPixelBuffer? {
    let width = src.width
    let height = src.height
    if width == 0 || height == 0 { return nil }

    // Downscale if either dimension exceeds maxSide (Vision works fine on
    // ~2K images and is much faster than on 4K+).
    let longest = max(width, height)
    var scale: CGFloat = 1.0
    if longest > maxSide {
        scale = CGFloat(maxSide) / CGFloat(longest)
    }
    let scaledW = max(1, Int((CGFloat(width) * scale).rounded()))
    let scaledH = max(1, Int((CGFloat(height) * scale).rounded()))

    // Round UP to multiple of `align` — padding goes to right + bottom as
    // white pixels so we never crop content.
    let padded = { (n: Int) -> Int in
        let r = n % align
        return r == 0 ? n : n + (align - r)
    }
    let targetW = padded(scaledW)
    let targetH = padded(scaledH)

    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        targetW,
        targetH,
        kCVPixelFormatType_32BGRA,
        attrs as CFDictionary,
        &pb
    )
    guard status == kCVReturnSuccess, let buffer = pb else {
        FileHandle.standardError.write(
            "CVPixelBufferCreate failed status=\(status) size=\(targetW)x\(targetH)\n".data(using: .utf8)!
        )
        return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    let baseAddr = CVPixelBufferGetBaseAddress(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGImageAlphaInfo.noneSkipFirst.rawValue
        | CGBitmapInfo.byteOrder32Little.rawValue
    guard let ctx = CGContext(
        data: baseAddr,
        width: targetW,
        height: targetH,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: cs,
        bitmapInfo: info
    ) else { return nil }

    ctx.setFillColor(CGColor.white)
    ctx.fill(CGRect(x: 0, y: 0, width: targetW, height: targetH))
    // Anchor at top-left in CG y-up coordinates — padding lands on right/bottom.
    ctx.draw(src, in: CGRect(x: 0, y: targetH - scaledH, width: scaledW, height: scaledH))
    return buffer
}

func loadCGImage(_ path: String) -> CGImage? {
    if let img = NSImage(contentsOfFile: path) {
        var r = NSRect(origin: .zero, size: img.size)
        if let cg = img.cgImage(forProposedRect: &r, context: nil, hints: nil) {
            return cg
        }
    }
    // Fallback: CGImageSource (handles formats NSImage misses).
    let url = URL(fileURLWithPath: path) as CFURL
    if let src = CGImageSourceCreateWithURL(url, nil),
       CGImageSourceGetCount(src) > 0,
       let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) {
        return cg
    }
    return nil
}

func ocr(_ path: String) async -> String? {
    guard let raw = loadCGImage(path) else {
        FileHandle.standardError.write("loadCGImage nil for \(path)\n".data(using: .utf8)!)
        return nil
    }
    guard let pb = renderToPixelBuffer(raw) else {
        FileHandle.standardError.write(
            "renderToPixelBuffer nil for \(path) (\(raw.width)x\(raw.height))\n".data(using: .utf8)!
        )
        return nil
    }
    var req = RecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.recognitionLanguages = [Locale.Language(identifier: "en-US")]
    req.usesLanguageCorrection = true
    do {
        let observations = try await req.perform(on: pb)
        let lines = observations.compactMap { $0.topCandidates(1).first?.string }
        return lines.joined(separator: "\n")
    } catch {
        FileHandle.standardError.write(
            "ocr error for \(path): \(error)\n".data(using: .utf8)!
        )
        return nil
    }
}

@main
struct VisionOCR {
    static func main() async {
        guard CommandLine.arguments.count >= 2 else {
            FileHandle.standardError.write(
                "usage: vision_ocr_swift <path> [<path> ...]\n".data(using: .utf8)!
            )
            exit(2)
        }
        for path in CommandLine.arguments.dropFirst() {
            if let text = await ocr(path) {
                print(text)
                print("---END---")
            } else {
                print("---ERR---")
            }
        }
    }
}
