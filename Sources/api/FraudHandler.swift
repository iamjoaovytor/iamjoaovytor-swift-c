import Domain
import Foundation
import NIOCore
import NIOHTTP1

final class FraudHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    // IOData bypasses HTTPResponseEncoder — write pre-serialized bytes directly to the socket.
    typealias OutboundOut = IOData

    // Full HTTP responses pre-serialized: status line + headers + body in one buffer.
    // Eliminates HTTPResponseEncoder overhead (header iteration + 3 write() calls → 1).
    private static let prebuiltKeepAlive: [ByteBuffer] = {
        let alloc = ByteBufferAllocator()
        return FraudScoring.responseBodies.map { body in
            let header = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: \(body.utf8.count)\r\n\r\n"
            var buf = alloc.buffer(capacity: header.utf8.count + body.utf8.count)
            buf.writeString(header)
            buf.writeString(body)
            return buf
        }
    }()

    private static let prebuiltClose: [ByteBuffer] = {
        let alloc = ByteBufferAllocator()
        return FraudScoring.responseBodies.map { body in
            let header = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: \(body.utf8.count)\r\nconnection: close\r\n\r\n"
            var buf = alloc.buffer(capacity: header.utf8.count + body.utf8.count)
            buf.writeString(header)
            buf.writeString(body)
            return buf
        }
    }()

    private static let readyOK: ByteBuffer = {
        let s = "HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n"
        var buf = ByteBufferAllocator().buffer(capacity: s.utf8.count)
        buf.writeString(s)
        return buf
    }()

    private static let readyNotReady: ByteBuffer = {
        let s = "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n"
        var buf = ByteBufferAllocator().buffer(capacity: s.utf8.count)
        buf.writeString(s)
        return buf
    }()

    private let state: LoaderState
    private let debugStats: DebugStatsCollector
    private var head: HTTPRequestHead?
    private var body: ByteBuffer?

    init(state: LoaderState, debugStats: DebugStatsCollector) {
        self.state = state
        self.debugStats = debugStats
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)
        switch part {
        case .head(let head):
            self.head = head
            self.body = nil
        case .body(var buffer):
            if self.body == nil {
                self.body = buffer
            } else {
                self.body!.writeBuffer(&buffer)
            }
        case .end:
            guard let head = self.head else { return }
            handle(context: context, head: head, body: self.body)
            self.head = nil
            self.body = nil
        }
    }

    private func handle(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer?) {
        let keepAlive = head.isKeepAlive
        switch (head.method, head.uri) {
        case (.POST, "/fraud-score"):
            handleFraud(context: context, body: body, keepAlive: keepAlive)
        case (.GET, "/ready"):
            let status: HTTPResponseStatus = state.isReady ? .ok : .serviceUnavailable
            writeEmpty(context: context, status: status, keepAlive: keepAlive)
        case (.GET, "/debug/stats"):
            do {
                let data = try debugStats.jsonData()
                var buf = context.channel.allocator.buffer(capacity: data.count)
                buf.writeBytes(data)
                writeJSON(context: context, status: .ok, buffer: buf, keepAlive: keepAlive)
            } catch {
                writeJSONString(context: context, status: .internalServerError, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            }
        case (.POST, "/debug/stats/reset"):
            debugStats.reset()
            writeJSONString(context: context, status: .ok, body: RinhaAPI.okBody, keepAlive: keepAlive)
        default:
            writeEmpty(context: context, status: .notFound, keepAlive: keepAlive)
        }
    }

    private func handleFraud(context: ChannelHandlerContext, body: ByteBuffer?, keepAlive: Bool) {
        guard let loaded = state.current, let body = body else {
            writeJSONString(context: context, status: .ok, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            return
        }
        var metrics = RequestPhaseMetrics()
        do {
            let quantized: [Int16]
            do {
                let parseStarted = DispatchTime.now().uptimeNanoseconds
                let parsed = try body.withUnsafeReadableBytes { rawBuffer in
                    try FastRequestParser.parsedQuery(from: rawBuffer)
                }
                metrics.parseNs = DispatchTime.now().uptimeNanoseconds - parseStarted
                let vectorizeStarted = DispatchTime.now().uptimeNanoseconds
                quantized = loaded.vectorizer.quantize(
                    transactionAmount: parsed.transactionAmount,
                    installments: parsed.installments,
                    requestedAt: parsed.requestedAt,
                    customerAvgAmount: parsed.customerAvgAmount,
                    customerTxCount24h: parsed.customerTxCount24h,
                    knownMerchant: parsed.knownMerchant,
                    merchantAvgAmount: parsed.merchantAvgAmount,
                    terminalIsOnline: parsed.terminalIsOnline,
                    terminalCardPresent: parsed.terminalCardPresent,
                    terminalKmFromHome: parsed.terminalKmFromHome,
                    merchantMccCode: parsed.merchantMccCode,
                    lastTransaction: parsed.lastTransaction
                )
                metrics.vectorizeNs = DispatchTime.now().uptimeNanoseconds - vectorizeStarted
                metrics.fastPath = true
            } catch {
                let bodyData = Data(body.readableBytesView)
                let parseStarted = DispatchTime.now().uptimeNanoseconds
                let fraudRequest = try RinhaAPI.decoder.decode(FraudRequest.self, from: bodyData)
                metrics.parseNs = DispatchTime.now().uptimeNanoseconds - parseStarted
                let vectorizeStarted = DispatchTime.now().uptimeNanoseconds
                let raw = try loaded.vectorizer.vectorize(fraudRequest)
                quantized = loaded.vectorizer.quantize(raw)
                metrics.vectorizeNs = DispatchTime.now().uptimeNanoseconds - vectorizeStarted
                metrics.fallbackPath = true
            }
            let searchStarted = DispatchTime.now().uptimeNanoseconds
            let rawFraudVotes: Int
            if debugStats.isEnabled {
                var searchMetrics = SearchMetrics()
                rawFraudVotes = KNN.fraudVoteCount(
                    query: quantized,
                    in: loaded.index,
                    ivf: loaded.ivf,
                    pq: loaded.pq,
                    pkey: loaded.pkey,
                    config: loaded.searchConfig,
                    metrics: &searchMetrics,
                    k: 5
                )
                metrics.searchCentroidNs = searchMetrics.centroidSearchNs
                metrics.searchShortlistNs = searchMetrics.shortlistNs
                metrics.searchExactFallbackCount = searchMetrics.exactFallbackCount
                metrics.searchAdaptiveExpandCount = searchMetrics.adaptiveExpandCount
            } else {
                rawFraudVotes = KNN.fraudVoteCount(
                    query: quantized,
                    in: loaded.index,
                    ivf: loaded.ivf,
                    pq: loaded.pq,
                    pkey: loaded.pkey,
                    config: loaded.searchConfig,
                    k: 5
                )
            }
            metrics.searchNs = DispatchTime.now().uptimeNanoseconds - searchStarted
            let responseStarted = DispatchTime.now().uptimeNanoseconds
            writeFraudResponse(context: context, votes: rawFraudVotes, keepAlive: keepAlive)
            metrics.responseNs = DispatchTime.now().uptimeNanoseconds - responseStarted
            debugStats.record(metrics)
        } catch {
            metrics.failed = true
            writeJSONString(context: context, status: .ok, body: RinhaAPI.fallbackBody, keepAlive: keepAlive)
            debugStats.record(metrics)
        }
    }

    private func writeFraudResponse(context: ChannelHandlerContext, votes: Int, keepAlive: Bool) {
        let buf = keepAlive ? FraudHandler.prebuiltKeepAlive[votes] : FraudHandler.prebuiltClose[votes]
        if keepAlive {
            context.writeAndFlush(wrapOutboundOut(.byteBuffer(buf)), promise: nil)
        } else {
            let promise = context.eventLoop.makePromise(of: Void.self)
            context.writeAndFlush(wrapOutboundOut(.byteBuffer(buf)), promise: promise)
            promise.futureResult.whenComplete { _ in context.close(promise: nil) }
        }
    }

    private func writeRawBuffer(context: ChannelHandlerContext, buf: ByteBuffer, keepAlive: Bool) {
        if keepAlive {
            context.writeAndFlush(wrapOutboundOut(.byteBuffer(buf)), promise: nil)
        } else {
            let promise = context.eventLoop.makePromise(of: Void.self)
            context.writeAndFlush(wrapOutboundOut(.byteBuffer(buf)), promise: promise)
            promise.futureResult.whenComplete { _ in context.close(promise: nil) }
        }
    }

    private func writeJSONString(context: ChannelHandlerContext, status: HTTPResponseStatus, body: String, keepAlive: Bool) {
        let close = keepAlive ? "" : "connection: close\r\n"
        let s = "HTTP/1.1 \(status.code) \(status.reasonPhrase)\r\ncontent-type: application/json\r\ncontent-length: \(body.utf8.count)\r\n\(close)\r\n\(body)"
        var buf = context.channel.allocator.buffer(capacity: s.utf8.count)
        buf.writeString(s)
        writeRawBuffer(context: context, buf: buf, keepAlive: keepAlive)
    }

    private func writeJSON(context: ChannelHandlerContext, status: HTTPResponseStatus, buffer: ByteBuffer, keepAlive: Bool) {
        let close = keepAlive ? "" : "connection: close\r\n"
        let header = "HTTP/1.1 \(status.code) \(status.reasonPhrase)\r\ncontent-type: application/json\r\ncontent-length: \(buffer.readableBytes)\r\n\(close)\r\n"
        var buf = context.channel.allocator.buffer(capacity: header.utf8.count + buffer.readableBytes)
        buf.writeString(header)
        var body = buffer
        buf.writeBuffer(&body)
        writeRawBuffer(context: context, buf: buf, keepAlive: keepAlive)
    }

    private func writeEmpty(context: ChannelHandlerContext, status: HTTPResponseStatus, keepAlive: Bool) {
        if status == .ok {
            writeRawBuffer(context: context, buf: FraudHandler.readyOK, keepAlive: keepAlive)
        } else if status == .serviceUnavailable {
            writeRawBuffer(context: context, buf: FraudHandler.readyNotReady, keepAlive: keepAlive)
        } else {
            let s = "HTTP/1.1 \(status.code) \(status.reasonPhrase)\r\ncontent-length: 0\r\n\r\n"
            var buf = context.channel.allocator.buffer(capacity: s.utf8.count)
            buf.writeString(s)
            writeRawBuffer(context: context, buf: buf, keepAlive: keepAlive)
        }
    }
}
