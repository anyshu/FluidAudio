import FluidAudio
import Foundation

/// Configuration for the remote Qwen3-ASR endpoint.
///
/// Mirrors the parameters that `Examples/Qwen3RemoteBenchmark/benchmark.py`
/// exposes, scoped down to what the AudioClaw demo needs at runtime.
struct Qwen3RemoteConfig: Equatable, Sendable {
    /// OpenAI-compatible chat-completions URL. Defaults to the public
    /// preview endpoint documented in `benchmarks.md`.
    var endpoint: URL
    /// Model identifier sent in the request body.
    var model: String
    /// Bearer token. Empty string means "no auth header" (will fail on the
    /// public endpoint, but allows users to point at a private server).
    var apiKey: String
    /// Per-request timeout. The public endpoint can spend several seconds
    /// on a single 10-30s utterance.
    var timeout: TimeInterval

    static let `default` = Qwen3RemoteConfig(
        endpoint: URL(string: "https://next-api.fazhiplus.com/v1/chat/completions")!,
        model: "Qwen3-ASR-1.7B",
        apiKey: "",
        timeout: 60
    )
}

enum Qwen3RemoteError: LocalizedError {
    case missingApiKey
    case invalidEndpoint
    case http(status: Int, body: String)
    case malformedResponse(String)
    case wavEncodingFailed

    var errorDescription: String? {
        switch self {
        case .missingApiKey:
            return
                "Remote Qwen3-ASR requires an API key. Open the model settings and paste your key (or set QWEN3_API_KEY in the environment)."
        case .invalidEndpoint:
            return "The configured Qwen3-ASR endpoint URL is invalid."
        case .http(let status, let body):
            let preview = body.prefix(200)
            return "Qwen3-ASR endpoint returned HTTP \(status). \(preview)"
        case .malformedResponse(let detail):
            return "Could not parse Qwen3-ASR response: \(detail)"
        case .wavEncodingFailed:
            return "Failed to encode captured audio as WAV before upload."
        }
    }
}

/// Sends 16 kHz mono Float32 samples to an OpenAI-compatible chat-completions
/// endpoint that fronts Qwen3-ASR-1.7B, parses the transcript out of the
/// response, and strips the `language <Lang><asr_text>...</asr_text>` wrapper.
///
/// Drop-in replacement (signature-wise) for the offline `transcribe(samples:)`
/// pattern used by the rest of `AsrController`.
actor Qwen3RemoteClient {
    private var config: Qwen3RemoteConfig
    private let session: URLSession

    init(config: Qwen3RemoteConfig) {
        self.config = config

        // Use a non-shared session so the per-request timeout matches the
        // configured value without leaking onto unrelated URLSession traffic.
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = config.timeout
        sessionConfig.timeoutIntervalForResource = config.timeout * 2
        sessionConfig.waitsForConnectivity = false
        self.session = URLSession(configuration: sessionConfig)
    }

    /// Update auth/endpoint/model without rebuilding the client. The HTTP
    /// session keeps its timeout from `init`; for runtime timeout changes,
    /// rebuild the client.
    func updateConfig(_ newConfig: Qwen3RemoteConfig) {
        self.config = newConfig
    }

    /// Transcribe 16 kHz mono Float32 samples. The samples are encoded as a
    /// 16-bit PCM WAV inlined into the request as a `data:audio/wav;base64,…`
    /// URL — matching the Python benchmark exactly so behavior is parity.
    func transcribe(samples: [Float]) async throws -> String {
        guard !config.apiKey.isEmpty else { throw Qwen3RemoteError.missingApiKey }

        let wavData = try wavData(from: samples)
        let dataURL = "data:audio/wav;base64,\(wavData.base64EncodedString())"

        var request = URLRequest(url: config.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = try buildRequestBody(dataURL: dataURL)

        return try await sendWithRetries(request: request)
    }

    // MARK: - Networking

    /// Mirrors `call_with_retries` in benchmark.py: retries on 429/500/502/503/504
    /// with exponential backoff (1s → 30s, 4 retries).
    private func sendWithRetries(request: URLRequest) async throws -> String {
        var delaySeconds: Double = 1.0
        let maxRetries = 4

        for attempt in 0...maxRetries {
            do {
                return try await sendOnce(request: request)
            } catch let Qwen3RemoteError.http(status, body) where Self.isTransient(status) {
                if attempt == maxRetries {
                    throw Qwen3RemoteError.http(status: status, body: body)
                }
                try await Task.sleep(nanoseconds: UInt64(delaySeconds * 1_000_000_000))
                delaySeconds = min(delaySeconds * 2, 30)
            }
        }
        // Unreachable: the loop either returns or throws on the last attempt.
        throw Qwen3RemoteError.malformedResponse("retry loop exited unexpectedly")
    }

    private func sendOnce(request: URLRequest) async throws -> String {
        let (data, response) = try await session.data(for: request)

        guard let http = response as? HTTPURLResponse else {
            throw Qwen3RemoteError.malformedResponse("response was not HTTPURLResponse")
        }

        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "<binary>"
            throw Qwen3RemoteError.http(status: http.statusCode, body: body)
        }

        return try parseTranscript(from: data)
    }

    private static func isTransient(_ status: Int) -> Bool {
        status == 429 || (500...504).contains(status)
    }

    // MARK: - Request body

    /// Builds the `messages` payload. Sends only the audio block as the user
    /// content — the public endpoint returns the transcript even with no
    /// explicit text prompt, and adding one (e.g. "Please transcribe.")
    /// occasionally makes the model echo it back.
    private func buildRequestBody(dataURL: String) throws -> Data {
        let body: [String: Any] = [
            "model": config.model,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "audio_url",
                            "audio_url": ["url": dataURL],
                        ]
                    ],
                ]
            ],
        ]
        return try JSONSerialization.data(withJSONObject: body, options: [])
    }

    // MARK: - Response parsing

    /// Pulls `choices[0].message.content` out of the OpenAI-style payload and
    /// strips the `language <Lang><asr_text>…</asr_text>` envelope.
    /// Some providers wrap content as a list of typed blocks; both forms are
    /// handled.
    private func parseTranscript(from data: Data) throws -> String {
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = json["choices"] as? [[String: Any]],
            let first = choices.first,
            let message = first["message"] as? [String: Any]
        else {
            throw Qwen3RemoteError.malformedResponse("missing choices[0].message")
        }

        let rawContent: String
        if let str = message["content"] as? String {
            rawContent = str
        } else if let blocks = message["content"] as? [[String: Any]] {
            rawContent = blocks.compactMap { ($0["text"] as? String) }.joined()
        } else {
            throw Qwen3RemoteError.malformedResponse("unexpected content shape")
        }

        return Self.stripQwen3AsrTags(rawContent)
    }

    /// Matches benchmark.py's `strip_qwen3_asr_tags`: removes the leading
    /// `language <Lang><asr_text>` marker and any closing `</asr_text>` tag.
    static func stripQwen3AsrTags(_ text: String) -> String {
        var result = text
        // Leading `language <Lang><asr_text>` (case-insensitive).
        if let openMatch = result.range(
            of: #"^\s*language\s+\S+\s*<asr_text>"#,
            options: [.regularExpression, .caseInsensitive])
        {
            result.removeSubrange(openMatch)
        }
        // Trailing `</asr_text>`.
        if let closeMatch = result.range(
            of: #"</asr_text>\s*$"#,
            options: [.regularExpression, .caseInsensitive])
        {
            result.removeSubrange(closeMatch)
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - WAV encoding

    /// Reuses FluidAudio's `AudioWAV.data(from:sampleRate:)` which already
    /// emits 16-bit PCM WAV — exactly what `audio_to_data_url` in the Python
    /// benchmark produces.
    private func wavData(from samples: [Float]) throws -> Data {
        do {
            return try AudioWAV.data(from: samples, sampleRate: 16_000)
        } catch {
            throw Qwen3RemoteError.wavEncodingFailed
        }
    }
}
