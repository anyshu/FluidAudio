import FluidAudio
import Foundation

/// Configuration for the remote VibeVoice-ASR-7B endpoint. Mirrors the knobs
/// exposed by `Examples/VibeVoiceBenchmark/benchmark.py`.
struct VibeVoiceRemoteConfig: Equatable, Sendable {
    var endpoint: URL
    var model: String
    var apiKey: String
    /// VibeVoice can take a while for long-form audio. Benchmark default = 600s.
    var timeout: TimeInterval
    /// `max_tokens` budget for the JSON diarization output. 32768 is the
    /// benchmark default but for AudioClaw's VAD-segmented chunks 2048 is
    /// usually plenty and finishes faster.
    var maxTokens: Int
    var systemPrompt: String
    /// User-prompt template with `{duration}` placeholder (seconds, %.2f).
    var userPromptTemplate: String

    static let `default` = VibeVoiceRemoteConfig(
        endpoint: URL(string: "https://next-api.fazhiplus.com/v1/chat/completions")!,
        model: "VibeVoice-ASR-7B",
        apiKey: "",
        timeout: 600,
        maxTokens: 2048,
        systemPrompt:
            "You are a helpful assistant that transcribes audio input into text output in JSON format.",
        userPromptTemplate:
            "This is a {duration} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
    )
}

enum VibeVoiceRemoteError: LocalizedError {
    case missingApiKey
    case http(status: Int, body: String)
    case malformedResponse(String)
    case wavEncodingFailed

    var errorDescription: String? {
        switch self {
        case .missingApiKey:
            return
                "Remote VibeVoice-ASR requires an API key. Open the model panel and paste your key (or set QWEN3_API_KEY / OPENAI_API_KEY in the environment)."
        case .http(let status, let body):
            let preview = body.prefix(200)
            return "VibeVoice endpoint returned HTTP \(status). \(preview)"
        case .malformedResponse(let detail):
            return "Could not parse VibeVoice response: \(detail)"
        case .wavEncodingFailed:
            return "Failed to encode captured audio as WAV before upload."
        }
    }
}

/// Sends 16 kHz mono Float32 samples to an OpenAI-compatible chat-completions
/// endpoint fronting VibeVoice-ASR-7B. Parses the JSON diarization payload
/// the model returns and concatenates the `Content` fields into one transcript
/// — matching `extract_transcript` in benchmark.py.
actor VibeVoiceRemoteClient {
    private var config: VibeVoiceRemoteConfig
    private let session: URLSession

    init(config: VibeVoiceRemoteConfig) {
        self.config = config

        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = config.timeout
        sessionConfig.timeoutIntervalForResource = config.timeout * 2
        sessionConfig.waitsForConnectivity = false
        self.session = URLSession(configuration: sessionConfig)
    }

    func updateConfig(_ newConfig: VibeVoiceRemoteConfig) {
        self.config = newConfig
    }

    /// Transcribe 16 kHz mono Float32 samples by uploading them as a
    /// base64-WAV data URL. Returns the concatenated transcript.
    func transcribe(samples: [Float]) async throws -> String {
        guard !config.apiKey.isEmpty else { throw VibeVoiceRemoteError.missingApiKey }

        let wavData = try wavData(from: samples)
        let dataURL = "data:audio/wav;base64,\(wavData.base64EncodedString())"
        let duration = Double(samples.count) / 16_000.0

        var request = URLRequest(url: config.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try buildRequestBody(dataURL: dataURL, duration: duration)

        let rawContent = try await sendWithRetries(request: request)
        return Self.extractTranscript(from: rawContent)
    }

    // MARK: - Networking

    private func sendWithRetries(request: URLRequest) async throws -> String {
        var delaySeconds: Double = 1.0
        let maxRetries = 4

        for attempt in 0...maxRetries {
            do {
                return try await sendOnce(request: request)
            } catch let VibeVoiceRemoteError.http(status, body) where Self.isTransient(status) {
                if attempt == maxRetries {
                    throw VibeVoiceRemoteError.http(status: status, body: body)
                }
                try await Task.sleep(nanoseconds: UInt64(delaySeconds * 1_000_000_000))
                delaySeconds = min(delaySeconds * 2, 30)
            }
        }
        throw VibeVoiceRemoteError.malformedResponse("retry loop exited unexpectedly")
    }

    private func sendOnce(request: URLRequest) async throws -> String {
        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw VibeVoiceRemoteError.malformedResponse("response was not HTTPURLResponse")
        }

        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "<binary>"
            throw VibeVoiceRemoteError.http(status: http.statusCode, body: body)
        }

        return try parseAssistantContent(from: data)
    }

    private static func isTransient(_ status: Int) -> Bool {
        status == 429 || (500...504).contains(status)
    }

    // MARK: - Request body

    /// Body shape:
    /// ```json
    /// {
    ///   "model": "VibeVoice-ASR-7B",
    ///   "messages": [
    ///     {"role": "system", "content": "<system prompt>"},
    ///     {"role": "user", "content": [
    ///       {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}},
    ///       {"type": "text", "text": "<duration-aware user prompt>"}
    ///     ]}
    ///   ],
    ///   "max_tokens": 2048,
    ///   "temperature": 0.0,
    ///   "stream": false,
    ///   "top_p": 1.0
    /// }
    /// ```
    private func buildRequestBody(dataURL: String, duration: Double) throws -> Data {
        let userPrompt = config.userPromptTemplate.replacingOccurrences(
            of: "{duration}", with: String(format: "%.2f", duration))

        let body: [String: Any] = [
            "model": config.model,
            "messages": [
                ["role": "system", "content": config.systemPrompt],
                [
                    "role": "user",
                    "content": [
                        ["type": "audio_url", "audio_url": ["url": dataURL]],
                        ["type": "text", "text": userPrompt],
                    ],
                ],
            ],
            "max_tokens": config.maxTokens,
            "temperature": 0.0,
            "stream": false,
            "top_p": 1.0,
        ]
        return try JSONSerialization.data(withJSONObject: body, options: [])
    }

    // MARK: - Response parsing

    /// Pulls `choices[0].message.content` out of the OpenAI payload. The
    /// content is a JSON string that callers parse with `extractTranscript`.
    private func parseAssistantContent(from data: Data) throws -> String {
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = json["choices"] as? [[String: Any]],
            let first = choices.first,
            let message = first["message"] as? [String: Any]
        else {
            throw VibeVoiceRemoteError.malformedResponse("missing choices[0].message")
        }

        if let str = message["content"] as? String { return str }
        if let blocks = message["content"] as? [[String: Any]] {
            return blocks.compactMap { ($0["text"] as? String) }.joined()
        }
        throw VibeVoiceRemoteError.malformedResponse("unexpected content shape")
    }

    /// Port of `extract_transcript` in benchmark.py.
    ///
    /// VibeVoice answers with a JSON array shaped like
    /// `[{"Start time": "...", "End time": "...", "Speaker ID": "...", "Content": "..."}]`,
    /// sometimes wrapped in markdown ```json fences or surrounded by prose.
    /// Concatenate every `Content` value in order, separated by single spaces.
    /// Drops `[silence]` / `[noise]` / `[music]` / `[inaudible]` placeholders
    /// and squashes consecutive duplicates (the model loves to loop once it
    /// has nothing left to say).
    static func extractTranscript(from raw: String) -> String {
        let unwrapped = stripMarkdownFence(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        if unwrapped.isEmpty { return "" }

        // 1. Direct JSON parse → array of dicts.
        if let segments = parseSegmentArray(from: unwrapped) {
            return collapseSegments(segments)
        }
        // 2. Extract the first `[ ... ]` substring and try again.
        if let arrayRange = unwrapped.range(of: #"\[[\s\S]*?\]"#, options: .regularExpression),
            let segments = parseSegmentArray(from: String(unwrapped[arrayRange]))
        {
            return collapseSegments(segments)
        }
        // 3. Last-resort: scrape `Content: …` lines from arbitrary text.
        let contentLines = scrapeContentLines(from: unwrapped)
        if !contentLines.isEmpty {
            return collapseSegments(contentLines.map { ["Content": $0] })
        }
        // 4. Fall back to the raw payload.
        return unwrapped
    }

    private static func stripMarkdownFence(_ text: String) -> String {
        var s = text
        // Leading ``` or ```json
        if let openFence = s.range(
            of: #"^\s*```(?:json|JSON)?\s*"#,
            options: [.regularExpression])
        {
            s.removeSubrange(openFence)
        }
        // Trailing ```
        if let closeFence = s.range(
            of: #"\s*```\s*$"#,
            options: [.regularExpression])
        {
            s.removeSubrange(closeFence)
        }
        return s
    }

    private static func parseSegmentArray(from text: String) -> [[String: Any]]? {
        guard let data = text.data(using: .utf8) else { return nil }
        let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        if let array = parsed as? [[String: Any]] { return array }
        if let dict = parsed as? [String: Any] { return [dict] }
        return nil
    }

    /// Concatenate `Content` (case-insensitive) fields, dropping noise tokens
    /// and consecutive duplicates.
    private static func collapseSegments(_ segments: [[String: Any]]) -> String {
        let placeholderRegex = try? NSRegularExpression(
            pattern: #"^\s*\[\s*(silence|noise|music|inaudible)\s*\]\s*$"#,
            options: [.caseInsensitive])

        var contents: [String] = []
        for seg in segments {
            let raw =
                (seg["Content"] as? String) ?? (seg["content"] as? String)
                ?? (seg["text"] as? String) ?? ""
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if value.isEmpty { continue }
            if let regex = placeholderRegex {
                let range = NSRange(value.startIndex..<value.endIndex, in: value)
                if regex.firstMatch(in: value, options: [], range: range) != nil { continue }
            }
            if contents.last == value { continue }
            contents.append(value)
        }
        return contents.joined(separator: " ")
    }

    /// Fallback: pull `Content: ...` or `"Content": "..."` lines out of free text.
    private static func scrapeContentLines(from text: String) -> [String] {
        let pattern = #"(?im)^[\s\-\*]*"?Content"?\s*[:：]\s*"?(.+?)"?\s*[,}]?\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }
        let nsText = text as NSString
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length))
        return matches.compactMap { match in
            guard match.numberOfRanges >= 2 else { return nil }
            let captured = nsText.substring(with: match.range(at: 1))
            return captured.trimmingCharacters(in: CharacterSet(charactersIn: "\" ,"))
        }
    }

    // MARK: - WAV encoding

    private func wavData(from samples: [Float]) throws -> Data {
        do {
            return try AudioWAV.data(from: samples, sampleRate: 16_000)
        } catch {
            throw VibeVoiceRemoteError.wavEncodingFailed
        }
    }
}
