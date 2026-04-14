import AppKit
import SwiftUI

struct TranscriptTextView: NSViewRepresentable {
    let text: String

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        scrollView.autohidesScrollers = true

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 0, height: 8)
        textView.font = .systemFont(ofSize: 14)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        let previousText = context.coordinator.lastText
        guard previousText != text else { return }

        let wasNearBottom = scrollView.isNearBottom
        if text.hasPrefix(previousText), let storage = textView.textStorage {
            let delta = String(text.dropFirst(previousText.count))
            if !delta.isEmpty {
                storage.append(NSAttributedString(string: delta))
            }
        } else {
            textView.string = text
        }

        context.coordinator.lastText = text
        if wasNearBottom || previousText.isEmpty {
            textView.scrollToEndOfDocument(nil)
        }
    }

    final class Coordinator {
        var lastText: String = ""
    }
}

private extension NSScrollView {
    var isNearBottom: Bool {
        guard let contentView = documentView else { return true }
        let visibleMaxY = self.contentView.documentVisibleRect.maxY
        let contentHeight = contentView.frame.maxY
        return contentHeight - visibleMaxY < 48
    }
}
