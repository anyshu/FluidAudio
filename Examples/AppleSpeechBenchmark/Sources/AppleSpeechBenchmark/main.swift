import AVFoundation
import Foundation
import Speech

// MARK: - CLI argument parsing

struct BenchmarkOptions {
    var dataset: String = "all"          // thchs30 | librispeech | librispeech-other | jsut | all
    var maxFiles: Int = Int.max
    var outputPath: String? = nil
    var onDeviceOnly: Bool = true
    var datasetsRoot: String =
        (NSHomeDirectory() as NSString).appendingPathComponent("Library/Application Support/FluidAudio/Datasets")
}

func parseArgs() -> BenchmarkOptions {
    var opts = BenchmarkOptions()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--dataset":
            if let v = it.next() { opts.dataset = v.lowercased() }
        case "--max-files":
            if let v = it.next(), let n = Int(v) { opts.maxFiles = n }
        case "--output":
            opts.outputPath = it.next()
        case "--datasets-root":
            if let v = it.next() { opts.datasetsRoot = v }
        case "--allow-server":
            opts.onDeviceOnly = false
        case "--verbose", "-v":
            verboseRecognition = true
        case "-h", "--help":
            printHelp()
            exit(0)
        default:
            FileHandle.standardError.write(Data("Unknown arg: \(arg)\n".utf8))
            exit(2)
        }
    }
    return opts
}

func printHelp() {
    print(
        """
        Apple SFSpeechRecognizer benchmark

        Usage:
          AppleSpeechBenchmark [options]

        Options:
          --dataset <name>       thchs30 | librispeech | librispeech-other | jsut | all (default: all)
          --max-files <n>        Limit number of files per dataset (default: all)
          --output <path>        Save JSON to path. Default: <repo>/benchmark_results/apple_<dataset>.json
                                 (pass 'none' to skip writing)
          --allow-server         Allow server-side recognition (default: on-device only)
          --datasets-root <dir>  Override datasets root (default: ~/Library/Application Support/FluidAudio/Datasets)
          -h, --help             Show help
        """
    )
}

/// Walk up from the executable's directory until we find a sibling `Package.swift` whose
/// directory contains either a `Sources/FluidAudio/` (the main repo) or an existing
/// `benchmark_results/` directory. Returns the resolved repo root, or nil if not found.
func findRepoRoot() -> URL? {
    let exec = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    var dir = exec.deletingLastPathComponent()
    let fm = FileManager.default
    for _ in 0..<8 {
        let pkg = dir.appendingPathComponent("Package.swift")
        let mainSources = dir.appendingPathComponent("Sources/FluidAudio")
        let resultsDir = dir.appendingPathComponent("benchmark_results")
        if fm.fileExists(atPath: pkg.path)
            && (fm.fileExists(atPath: mainSources.path) || fm.fileExists(atPath: resultsDir.path))
        {
            return dir
        }
        let parent = dir.deletingLastPathComponent()
        if parent.path == dir.path { break }
        dir = parent
    }
    return nil
}

func defaultOutputPath(dataset: String) -> String {
    let suffix = dataset == "all" ? "all" : dataset.replacingOccurrences(of: "-", with: "_")
    let filename = "apple_\(suffix).json"
    if let root = findRepoRoot() {
        return root.appendingPathComponent("benchmark_results").appendingPathComponent(filename).path
    }
    // Fallback: cwd
    return filename
}

// MARK: - Text normalization (fair cross-benchmark comparison)

enum Metric { case wer, cer }

struct Normalizer {
    // Additional diacritic replacements mirrored from Swift TextNormalizer
    static let additionalDiacritics: [Character: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
        "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
    ]

    /// Simple normalization for English: lowercase, strip punctuation/symbols, collapse whitespace.
    /// Equivalent to HF "BasicTextNormalizer" (not the full EnglishTextNormalizer). Matches
    /// `basicNormalize` in the Swift codebase. For strict HF leaderboard WER, run the Swift
    /// CLI's TextNormalizer offline on both hypothesis and reference.
    static func english(_ text: String) -> String {
        var s = text.lowercased()
        // Remove bracketed content
        s = s.replacingOccurrences(of: "[<\\[].*?[>\\]]", with: "", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\([^)]+?\\)", with: "", options: .regularExpression)
        // NFKD
        s = s.precomposedStringWithCompatibilityMapping
        // Map symbols/punctuation/separators to space, drop combining marks
        var out = ""
        out.reserveCapacity(s.count)
        for ch in s {
            if let repl = additionalDiacritics[ch] {
                out.append(repl)
                continue
            }
            guard let scalar = ch.unicodeScalars.first else { continue }
            let cat = scalar.properties.generalCategory
            switch cat {
            case .nonspacingMark, .spacingMark, .enclosingMark:
                continue  // drop diacritics
            case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
                 .initialPunctuation, .finalPunctuation, .otherPunctuation,
                 .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
                 .spaceSeparator, .lineSeparator, .paragraphSeparator:
                out.append(" ")
            default:
                out.append(ch)
            }
        }
        out = out.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Chinese: keep CJK characters only, convert Arabic digits to Chinese characters
    /// (mirrors the Swift ctc-zh-cn benchmark and Python SenseVoice benchmark).
    static func chinese(_ text: String) -> String {
        let digitMap: [Character: Character] = [
            "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        ]
        var mapped = ""
        mapped.reserveCapacity(text.count)
        for ch in text {
            if let zh = digitMap[ch] {
                mapped.append(zh)
            } else {
                mapped.append(ch)
            }
        }
        // Keep only CJK unified ideographs
        var out = ""
        for scalar in mapped.unicodeScalars {
            let v = scalar.value
            if (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
                || (0x20000...0x2A6DF).contains(v)
            {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    /// Japanese: convert kanji numerals to Arabic digits, keep hiragana/katakana/CJK + digits.
    /// Order matches the Swift ja benchmark: compound tens before simple tens.
    static func japanese(_ text: String) -> String {
        let replacements: [(String, String)] = [
            ("二十一", "21"), ("二十二", "22"), ("二十三", "23"), ("二十四", "24"), ("二十五", "25"),
            ("二十六", "26"), ("二十七", "27"), ("二十八", "28"), ("二十九", "29"),
            ("三十一", "31"), ("三十二", "32"), ("三十三", "33"), ("三十四", "34"), ("三十五", "35"),
            ("三十六", "36"), ("三十七", "37"), ("三十八", "38"), ("三十九", "39"),
            ("四十一", "41"), ("四十二", "42"), ("四十三", "43"), ("四十四", "44"), ("四十五", "45"),
            ("四十六", "46"), ("四十七", "47"), ("四十八", "48"), ("四十九", "49"),
            ("五十一", "51"), ("五十二", "52"), ("五十三", "53"), ("五十四", "54"), ("五十五", "55"),
            ("五十六", "56"), ("五十七", "57"), ("五十八", "58"), ("五十九", "59"),
            ("六十一", "61"), ("六十二", "62"), ("六十三", "63"), ("六十四", "64"), ("六十五", "65"),
            ("六十六", "66"), ("六十七", "67"), ("六十八", "68"), ("六十九", "69"),
            ("七十一", "71"), ("七十二", "72"), ("七十三", "73"), ("七十四", "74"), ("七十五", "75"),
            ("七十六", "76"), ("七十七", "77"), ("七十八", "78"), ("七十九", "79"),
            ("八十一", "81"), ("八十二", "82"), ("八十三", "83"), ("八十四", "84"), ("八十五", "85"),
            ("八十六", "86"), ("八十七", "87"), ("八十八", "88"), ("八十九", "89"),
            ("九十一", "91"), ("九十二", "92"), ("九十三", "93"), ("九十四", "94"), ("九十五", "95"),
            ("九十六", "96"), ("九十七", "97"), ("九十八", "98"), ("九十九", "99"),
            ("十", "10"), ("二十", "20"), ("三十", "30"), ("四十", "40"), ("五十", "50"),
            ("六十", "60"), ("七十", "70"), ("八十", "80"), ("九十", "90"), ("百", "100"),
            ("千", "1000"), ("万", "10000"),
            ("一", "1"), ("二", "2"), ("三", "3"), ("四", "4"), ("五", "5"),
            ("六", "6"), ("七", "7"), ("八", "8"), ("九", "9"), ("零", "0"), ("〇", "0"),
        ]
        var s = text
        for (k, v) in replacements {
            s = s.replacingOccurrences(of: k, with: v)
        }
        // Keep only meaningful script characters + digits
        var out = ""
        for scalar in s.unicodeScalars {
            let v = scalar.value
            if (0x3040...0x309F).contains(v)    // hiragana
                || (0x30A0...0x30FF).contains(v)  // katakana
                || (0x4E00...0x9FFF).contains(v)  // CJK
                || (0x0030...0x0039).contains(v)  // digits
            {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }
}

// MARK: - Edit distance

func editDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
    if a.isEmpty { return b.count }
    if b.isEmpty { return a.count }
    var prev = Array(0...b.count)
    var curr = Array(repeating: 0, count: b.count + 1)
    for i in 1...a.count {
        curr[0] = i
        for j in 1...b.count {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1]
            } else {
                curr[j] = 1 + min(prev[j - 1], min(prev[j], curr[j - 1]))
            }
        }
        swap(&prev, &curr)
    }
    return prev[b.count]
}

func wer(reference: String, hypothesis: String) -> (errors: Int, refCount: Int) {
    let refTokens = reference.split(separator: " ").map(String.init)
    let hypTokens = hypothesis.split(separator: " ").map(String.init)
    return (editDistance(refTokens, hypTokens), refTokens.count)
}

func cer(reference: String, hypothesis: String) -> (errors: Int, refCount: Int) {
    let refChars = Array(reference)
    let hypChars = Array(hypothesis)
    return (editDistance(refChars, hypChars), refChars.count)
}

// MARK: - Audio duration

func audioDurationSeconds(url: URL) -> Double {
    guard let file = try? AVAudioFile(forReading: url) else { return 0 }
    let sr = file.processingFormat.sampleRate
    return sr > 0 ? Double(file.length) / sr : 0
}

// MARK: - Dataset loaders

struct Sample {
    let id: String
    let audioURL: URL
    let reference: String
}

struct DatasetSpec {
    let name: String
    let locale: Locale
    let metric: Metric
    let samples: [Sample]
}

enum DatasetError: Error { case missing(String), badMetadata(String) }

struct MetadataLine: Decodable {
    let file_name: String
    let text: String
}

func loadJSONL(root: URL) throws -> [Sample] {
    let meta = root.appendingPathComponent("metadata.jsonl")
    guard FileManager.default.fileExists(atPath: meta.path) else {
        throw DatasetError.missing(meta.path)
    }
    let text = try String(contentsOf: meta, encoding: .utf8)
    var samples: [Sample] = []
    for line in text.split(whereSeparator: { $0.isNewline }) {
        let data = Data(line.utf8)
        guard let entry = try? JSONDecoder().decode(MetadataLine.self, from: data) else { continue }
        let audioPath = entry.file_name.hasPrefix("audio/")
            ? root.appendingPathComponent(entry.file_name)
            : root.appendingPathComponent("audio").appendingPathComponent(entry.file_name)
        let id = (entry.file_name as NSString).lastPathComponent
        samples.append(Sample(id: id, audioURL: audioPath, reference: entry.text))
    }
    return samples
}

func loadLibriSpeech(root: URL) throws -> [Sample] {
    guard FileManager.default.fileExists(atPath: root.path) else {
        throw DatasetError.missing(root.path)
    }
    var samples: [Sample] = []
    let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
    while let url = enumerator?.nextObject() as? URL {
        guard url.pathExtension == "txt", url.lastPathComponent.hasSuffix(".trans.txt") else { continue }
        let content = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        for line in content.split(whereSeparator: { $0.isNewline }) {
            let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
            guard parts.count == 2 else { continue }
            let id = String(parts[0])
            let ref = String(parts[1])
            let flac = url.deletingLastPathComponent().appendingPathComponent("\(id).flac")
            if FileManager.default.fileExists(atPath: flac.path) {
                samples.append(Sample(id: id, audioURL: flac, reference: ref))
            }
        }
    }
    samples.sort { $0.id < $1.id }
    return samples
}

func resolveDataset(_ name: String, root: String) throws -> DatasetSpec {
    let rootURL = URL(fileURLWithPath: root)
    switch name {
    case "thchs30":
        let ds = rootURL.appendingPathComponent("THCHS-30")
        let samples = try loadJSONL(root: ds)
        return DatasetSpec(name: "THCHS-30", locale: Locale(identifier: "zh-CN"), metric: .cer, samples: samples)
    case "jsut":
        let ds = rootURL.appendingPathComponent("JSUT-basic5000")
        let samples = try loadJSONL(root: ds)
        return DatasetSpec(name: "JSUT-basic5000", locale: Locale(identifier: "ja-JP"), metric: .cer, samples: samples)
    case "librispeech":
        let ds = rootURL.appendingPathComponent("LibriSpeech/test-clean")
        let samples = try loadLibriSpeech(root: ds)
        return DatasetSpec(
            name: "LibriSpeech test-clean", locale: Locale(identifier: "en-US"), metric: .wer, samples: samples)
    case "librispeech-other":
        let ds = rootURL.appendingPathComponent("LibriSpeech/test-other")
        let samples = try loadLibriSpeech(root: ds)
        return DatasetSpec(
            name: "LibriSpeech test-other", locale: Locale(identifier: "en-US"), metric: .wer, samples: samples)
    default:
        throw DatasetError.badMetadata("Unknown dataset: \(name)")
    }
}

// MARK: - Recognition

enum RecognitionError: Error {
    case notAuthorized(SFSpeechRecognizerAuthorizationStatus)
    case recognizerUnavailable(String)
    case onDeviceUnavailable(String)
    case recognitionFailed(String)
    case timedOut
    case decodeFailed(String)
}

func requestAuth() async throws {
    print("Requesting Speech authorization...")
    let status: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { c in
        SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
    }
    let statusStr: String
    switch status {
    case .notDetermined: statusStr = "notDetermined"
    case .denied: statusStr = "denied"
    case .restricted: statusStr = "restricted"
    case .authorized: statusStr = "authorized"
    @unknown default: statusStr = "unknown"
    }
    print("Authorization status: \(statusStr)")
    guard status == .authorized else { throw RecognitionError.notAuthorized(status) }
}

/// Decode any supported audio file (wav/flac/mp3/…) into non-interleaved float PCM chunks.
/// Returns (durationSeconds, [AVAudioPCMBuffer]) for streaming into the recognizer.
func decodeAudio(url: URL) throws -> (Double, [AVAudioPCMBuffer]) {
    let file: AVAudioFile
    do {
        file = try AVAudioFile(forReading: url)
    } catch {
        throw RecognitionError.decodeFailed("open \(url.lastPathComponent): \(error.localizedDescription)")
    }
    let srcFormat = file.processingFormat
    let sampleRate = srcFormat.sampleRate
    let duration = sampleRate > 0 ? Double(file.length) / sampleRate : 0

    // SFSpeechAudioBufferRecognitionRequest expects the recognizer's nativeAudioFormat or a
    // format it can convert from. Float32 non-interleaved at the file's sample rate works well.
    let chunkFrames: AVAudioFrameCount = 16384
    var buffers: [AVAudioPCMBuffer] = []
    while file.framePosition < file.length {
        let remaining = AVAudioFrameCount(file.length - file.framePosition)
        let cap = min(chunkFrames, remaining)
        guard let buf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: cap) else {
            throw RecognitionError.decodeFailed("alloc buffer")
        }
        do {
            try file.read(into: buf, frameCount: cap)
        } catch {
            throw RecognitionError.decodeFailed("read: \(error.localizedDescription)")
        }
        if buf.frameLength == 0 { break }
        buffers.append(buf)
    }
    return (duration, buffers)
}

/// Recognize pre-decoded PCM buffers. Has a timeout so misbehaving files can't hang the run.
func recognize(
    buffers: [AVAudioPCMBuffer],
    recognizer: SFSpeechRecognizer,
    onDeviceOnly: Bool,
    timeoutSeconds: Double
) async throws -> String {
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = false
    if onDeviceOnly {
        request.requiresOnDeviceRecognition = true
    }
    if #available(macOS 13.0, *) {
        request.addsPunctuation = false
    }

    return try await withThrowingTaskGroup(of: String.self) { group in
        group.addTask {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<String, Error>) in
                let doneLock = NSLock()
                var done = false
                func finish(_ result: Result<String, Error>) {
                    doneLock.lock()
                    defer { doneLock.unlock() }
                    if done { return }
                    done = true
                    switch result {
                    case .success(let s): cont.resume(returning: s)
                    case .failure(let e): cont.resume(throwing: e)
                    }
                }
                if verboseRecognition {
                    print("    -> creating recognitionTask")
                }
                let task = recognizer.recognitionTask(with: request) { result, error in
                    if let error = error {
                        if verboseRecognition { print("    -> callback error: \(error)") }
                        finish(.failure(RecognitionError.recognitionFailed(error.localizedDescription)))
                        return
                    }
                    guard let result = result else {
                        if verboseRecognition { print("    -> callback nil result") }
                        return
                    }
                    if verboseRecognition {
                        print("    -> callback partial/final=\(result.isFinal) text=\"\(result.bestTranscription.formattedString)\"")
                    }
                    if result.isFinal {
                        finish(.success(result.bestTranscription.formattedString))
                    }
                }
                _ = task
                if verboseRecognition {
                    print("    -> appending \(buffers.count) buffers")
                }
                for buf in buffers {
                    request.append(buf)
                }
                request.endAudio()
                if verboseRecognition {
                    print("    -> endAudio() called")
                }
            }
        }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
            throw RecognitionError.timedOut
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

/// Global debug flag set from CLI.
var verboseRecognition = false

// MARK: - Result aggregation

struct FileResult {
    let id: String
    let audioDuration: Double
    let processingTime: Double
    let errors: Int
    let refCount: Int
    var score: Double { refCount == 0 ? 0 : Double(errors) / Double(refCount) * 100 }
    var rtfx: Double { processingTime > 0 ? audioDuration / processingTime : 0 }
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let idx = Int(Double(sorted.count - 1) * p)
    return sorted[idx]
}

func average(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    return values.reduce(0, +) / Double(values.count)
}

// MARK: - Main

func run() async {
    let opts = parseArgs()

    do {
        try await requestAuth()
    } catch {
        FileHandle.standardError.write(Data("Speech authorization failed: \(error)\n".utf8))
        exit(1)
    }

    let datasetNames: [String]
    switch opts.dataset {
    case "all": datasetNames = ["thchs30", "librispeech", "librispeech-other", "jsut"]
    default: datasetNames = [opts.dataset]
    }

    var allReports: [[String: Any]] = []

    for dsName in datasetNames {
        let spec: DatasetSpec
        do {
            spec = try resolveDataset(dsName, root: opts.datasetsRoot)
        } catch {
            print("Skipping \(dsName): \(error)")
            continue
        }

        guard let recognizer = SFSpeechRecognizer(locale: spec.locale) else {
            print("No recognizer for \(spec.locale.identifier); skipping \(spec.name)")
            continue
        }
        guard recognizer.isAvailable else {
            print("Recognizer for \(spec.locale.identifier) not available right now; skipping \(spec.name)")
            continue
        }
        if opts.onDeviceOnly && !recognizer.supportsOnDeviceRecognition {
            print(
                "On-device recognition unavailable for \(spec.locale.identifier). "
                    + "Install the offline language pack in System Settings > Keyboard > Dictation, "
                    + "or re-run with --allow-server."
            )
            continue
        }

        let samples = Array(spec.samples.prefix(opts.maxFiles))

        print("")
        print("=== \(spec.name) (\(spec.locale.identifier)) ===")
        print("Files to process: \(samples.count)")
        print("Metric: \(spec.metric == .wer ? "WER" : "CER")")
        print("On-device: \(opts.onDeviceOnly)")
        print("Recognizer supportsOnDeviceRecognition: \(recognizer.supportsOnDeviceRecognition)")
        print("Recognizer isAvailable: \(recognizer.isAvailable)")

        var results: [FileResult] = []
        var totalAudio = 0.0
        var totalProc = 0.0
        var failed = 0

        // Force verbose for the very first file so hangs are always diagnosable.
        let forceVerboseFirst = !verboseRecognition

        for (idx, sample) in samples.enumerated() {
            let savedVerbose = verboseRecognition
            if idx == 0 && forceVerboseFirst { verboseRecognition = true }
            if verboseRecognition {
                print("  [\(idx + 1)/\(samples.count)] decoding \(sample.audioURL.lastPathComponent)")
            }
            let dur: Double
            let buffers: [AVAudioPCMBuffer]
            do {
                (dur, buffers) = try decodeAudio(url: sample.audioURL)
            } catch {
                failed += 1
                if failed <= 5 {
                    print("  [\(idx + 1)/\(samples.count)] \(sample.id): decode FAILED \(error)")
                }
                continue
            }
            // Timeout: generous multiple of audio duration, with a floor. Most files on M-series
            // finish in ≤ real-time for SFSpeechRecognizer; we cap far above that to catch hangs.
            let timeout = max(30.0, dur * 10)
            let t0 = Date()
            let hypothesis: String
            do {
                hypothesis = try await recognize(
                    buffers: buffers, recognizer: recognizer, onDeviceOnly: opts.onDeviceOnly,
                    timeoutSeconds: timeout)
            } catch {
                failed += 1
                if failed <= 5 {
                    print("  [\(idx + 1)/\(samples.count)] \(sample.id): FAILED \(error)")
                }
                continue
            }
            let proc = Date().timeIntervalSince(t0)

            let (refNorm, hypNorm): (String, String)
            switch spec.metric {
            case .wer:
                refNorm = Normalizer.english(sample.reference)
                hypNorm = Normalizer.english(hypothesis)
            case .cer:
                if spec.locale.identifier.hasPrefix("zh") {
                    refNorm = Normalizer.chinese(sample.reference)
                    hypNorm = Normalizer.chinese(hypothesis)
                } else {
                    refNorm = Normalizer.japanese(sample.reference)
                    hypNorm = Normalizer.japanese(hypothesis)
                }
            }

            let score = spec.metric == .wer
                ? wer(reference: refNorm, hypothesis: hypNorm)
                : cer(reference: refNorm, hypothesis: hypNorm)

            let r = FileResult(
                id: sample.id, audioDuration: dur, processingTime: proc,
                errors: score.errors, refCount: score.refCount)
            results.append(r)
            totalAudio += dur
            totalProc += proc

            if idx == 0 && forceVerboseFirst { verboseRecognition = savedVerbose }
            if (idx + 1) % 10 == 0 || idx == samples.count - 1 || idx < 3 {
                let avgScore = average(results.map { $0.score })
                let avgRtfx = average(results.compactMap { $0.rtfx > 0 ? $0.rtfx : nil })
                print(
                    String(
                        format: "  [%d/%d] %@ %.1fs proc=%.2fs rtfx=%.2fx | avg %@=%.2f%% rtfx=%.2fx",
                        idx + 1, samples.count, sample.id, dur, proc, r.rtfx,
                        spec.metric == .wer ? "WER" : "CER", avgScore, avgRtfx))
            }
        }

        guard !results.isEmpty else {
            print("No successful recognitions for \(spec.name); failed=\(failed)")
            continue
        }

        let scores = results.map { $0.score }
        let rtfs = results.map { $0.rtfx }.filter { $0 > 0 }
        let metricName = spec.metric == .wer ? "WER" : "CER"
        let meanScore = average(scores)
        let medianScore = percentile(scores, 0.5)
        let medianRTFx = percentile(rtfs, 0.5)
        let overallRTFx = totalProc > 0 ? totalAudio / totalProc : 0

        let below5 = scores.filter { $0 < 5 }.count
        let below10 = scores.filter { $0 < 10 }.count
        let below20 = scores.filter { $0 < 20 }.count
        let n = results.count

        // Match SenseVoice benchmark output format exactly.
        print("")
        print("=== Benchmark Results ===")
        print("Dataset: \(spec.name)")
        print("Model: Apple SFSpeechRecognizer")
        print("Files processed: \(n)")
        print("")
        print(String(format: "Average %@: %.1f%%", metricName, meanScore))
        print(String(format: "Median %@: %.1f%%", metricName, medianScore))
        print(String(format: "Median RTFx: %.1fx", medianRTFx))
        print(
            String(
                format: "Overall RTFx: %.1fx (%.1fs / %.1fs)",
                overallRTFx, totalAudio, totalProc))
        print("")
        print("\(metricName) Distribution:")
        print(
            String(
                format: "  <5%%:  %d files (%.1f%%)", below5, Double(below5) / Double(n) * 100))
        print(
            String(
                format: "  <10%%: %d files (%.1f%%)", below10, Double(below10) / Double(n) * 100))
        print(
            String(
                format: "  <20%%: %d files (%.1f%%)", below20, Double(below20) / Double(n) * 100))
        if failed > 0 {
            print("")
            print("Note: \(failed) file(s) failed recognition and are excluded from the metrics above.")
        }

        allReports.append([
            "dataset": spec.name,
            "locale": spec.locale.identifier,
            "metric": metricName,
            "files_processed": n,
            "failed": failed,
            "average_score": meanScore,
            "median_score": medianScore,
            "audio_duration_s": totalAudio,
            "processing_time_s": totalProc,
            "overall_rtfx": overallRTFx,
            "median_rtfx": medianRTFx,
            "below_5pct": below5,
            "below_10pct": below10,
            "below_20pct": below20,
            "on_device_only": opts.onDeviceOnly,
        ])
    }

    // Resolve output path. `--output none` skips writing; default goes under
    // <repo>/benchmark_results/.
    let resolvedOutput: String?
    if let user = opts.outputPath {
        resolvedOutput = ["none", "no", "off", ""].contains(user.lowercased()) ? nil : user
    } else {
        resolvedOutput = defaultOutputPath(dataset: opts.dataset)
    }

    if let out = resolvedOutput {
        let payload: [String: Any] = [
            "model": "Apple SFSpeechRecognizer",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "reports": allReports,
        ]
        let outURL = URL(fileURLWithPath: out)
        try? FileManager.default.createDirectory(
            at: outURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) {
            do {
                try data.write(to: outURL)
                print("")
                print("Saved JSON: \(out)")
            } catch {
                print("")
                print("Failed to write JSON to \(out): \(error)")
            }
        }
    }
}

// IMPORTANT: SFSpeechRecognizer delivers its recognitionTask callbacks via the main
// run loop. If we block the main thread with a DispatchSemaphore the callback never
// fires and every recognition hangs forever. Run the main run loop instead and stop
// it from the async Task when work is done.
Task {
    await run()
    CFRunLoopStop(CFRunLoopGetMain())
}
CFRunLoopRun()
